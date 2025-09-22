import platform
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)  # 双保险
import matplotlib.pyplot as plt
# if 'mac' in platform.platform():
#     # macOS 下通常不需要特殊的渲染后端设置
#     pass
# else:
#     # 非 macOS：指定使用 EGL 作为 MuJoCo 的 GL 后端，以便无显示环境下渲染
#     os.environ['MUJOCO_GL'] = 'egl'
#     if 'SLURM_STEP_GPUS' in os.environ:
#         # 在 SLURM 作业环境下，将 EGL 使用的设备与分配的 GPU 对齐
#         os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
import yaml
import shutil
from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))

# NumPy compat shim for legacy Isaac Gym utils (np.float deprecation)
import numpy as _np
if not hasattr(_np, 'float'):
    _np.float = float  # type: ignore[attr-defined]

# Import Isaac Gym BEFORE torch to satisfy dependency order
from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
from tqdm import tqdm
import logging
import warnings

# logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=DeprecationWarning)  # 屏蔽冗余的弃用告警
import math
import torch.nn.functional as F
import json
import dataclasses
import tempfile
import typing as tp
from pathlib import Path
import imageio.v2 as imageio

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf

import torch
from dm_env import specs
from url_benchmark import utils
from url_benchmark import agent as agents
# from url_benchmark.logger import Logger
from gym import spaces
from url_benchmark.replay_buffer import DataBuffer
from url_benchmark.hilbert_dataset import HilbertRepreTestDataset
from url_benchmark.legged_gym_env_utils import build_isaac_namespace, _to_rgb_frame
from torch.utils.data import DataLoader
from collections import defaultdict
# pbar import
import datetime
from tqdm import tqdm


def merge_keys(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict):
            dict1[k] = merge_keys(dict1.get(k, {}), v)
        else:
            dict1[k] = v
    return dict1

@dataclasses.dataclass
class Config:
    """顶层配置

    - agent: 由 Hydra 实例化的 agent 配置（见 url_benchmark/agent 下实现）
    - 训练/评估/数据集等的所有关键参数
    """
    agent: tp.Any
    # misc
    run_group: str = "Debug"
    seed: int = 1
    device: str = "cuda"
    example_replay_path: str = "/root/workspace/HugWBC/dataset/example_trajectories"
    use_tb: bool = False
    use_wandb: bool = True
    # experiment
    experiment: str = "offline"
    # task settings
    task: str = "h1int"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    image_wh: int = 64
    action_repeat: int = 1
    discount: float = 0.98
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    p_currgoal: float = 0  # current goal ratio
    p_randomgoal: float = 0  # random goal ratio
    # env
    num_envs: int = 1
    episode_length_s: float = 10.0
    commands_resampling_time: float = 10.0
    headless: bool = True
    use_history_action: bool = True
    
    # eval
    num_eval_episodes: int = 10
    eval_every_steps: int = 20000
    num_final_eval_episodes: int = 50
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    # training
    num_grad_steps: int = 100000000
    dataset_ratio: float = 1.0
    log_every_steps: int = 1000
    num_seed_frames: int = 0
    replay_buffer_episodes: int = 5000  # 从离线缓冲区加载的 episode 数上限
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    goal_eval: bool = False
    # dataset
    load_replay_buffer: tp.Optional[str] = None
    # legged-gym dataset (Hilbert zarr) options
    # eval control
    eval_only: bool = False
    eval_all: bool = False
    save_video: bool = False
    resume_from: tp.Optional[str] = None
    resume_phi_from: tp.Optional[str] = None


ConfigStore.instance().store(name="workspace_config", node=Config)


def make_agent(
        obs_type: str, image_wh, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent, agents.DDPGAgent, agents.SFHumanoidAgent]:
    """利用 Hydra 配置实例化 agent，并注入观测/动作规格与探索步数。"""
    cfg.obs_type = obs_type
    cfg.image_wh = image_wh
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

class Workspace:
    """训练工作台，负责构建组件与承载训练/评估/保存等过程。"""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        assert not (cfg.headless and cfg.save_video), "headless and save_video cannot be both True"
        hydra_dir = Path.cwd()
        parent_dir = os.path.dirname(os.path.dirname(hydra_dir))
      
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                print(f"Falling back to cpu as {cfg.device} is not available")
                # logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)
        task = cfg.task
        self.domain = task.split('_', maxsplit=1)[0]

        exp_name = 'offline'
        if cfg.resume_from is not None:
            if self.cfg.resume_from.endswith('.pt'):
                self.work_dir = cfg.resume_from.split('models')[0].rstrip('/')
            else:
                self.work_dir = cfg.resume_from
            try:
                with open(os.path.join(self.work_dir, "wandb.yaml"), "r") as f:
                    wandb_run_id = yaml.safe_load(f)["run_id"]
            except:
                wandb_run_id = None
        else:
            self.work_dir = os.path.join(parent_dir, exp_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            wandb_run_id = None
        os.makedirs(self.work_dir, exist_ok=True)
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        # ===== Persist and restore cfg as YAML =====
        config_yaml_path = os.path.join(self.work_dir, "config.yaml")
        if cfg.resume_from is not None:
            # On resume, attempt to load the stored cfg and use it
            if os.path.exists(config_yaml_path):
                try:
                    with open(config_yaml_path, "r") as f:
                        loaded_cfg_dict = yaml.safe_load(f)
                    loaded_cfg = omgcf.OmegaConf.create(loaded_cfg_dict)
                    # Preserve current resume path and work_dir
                    loaded_cfg.resume_from = cfg.resume_from
                    loaded_cfg.work_dir = self.work_dir
                    loaded_cfg.eval_only = cfg.eval_only
                    loaded_cfg.device = cfg.device
                    loaded_cfg.agent.device = cfg.device
                    loaded_cfg.use_wandb = cfg.use_wandb
                    loaded_cfg['eval_all'] = cfg.eval_all
                    self.cfg = loaded_cfg
                    cfg = self.cfg
                    print("Loading cfg from", config_yaml_path)
                except Exception as e:
                    print(f"Warning: failed to load cfg from {config_yaml_path}: {e}. Using provided cfg.")
            else:
                print(f"Warning: {config_yaml_path} not found for resume. Using provided cfg.")
        else:
            # New run: save resolved cfg as YAML for future resume
            try:
                cfg_to_save = omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
                with open(config_yaml_path, "w") as f:
                    yaml.safe_dump(cfg_to_save, f, sort_keys=False)
            except Exception as e:
                print(f"Warning: failed to save cfg to {config_yaml_path}: {e}")
        if self.cfg.agent.mix_ratio > 0 and self.cfg.agent.random_sample_z:
            pass
        else:
            self.cfg.agent.mix_ratio = 0
            print("Masking mix_ratio to 0, because random_sample_z is False")
        exp_name += '_'.join([f"pr{str(self.cfg.p_randomgoal)}", f"phe{str(self.cfg.agent.hilp_expectile)}", f"phg{str(self.cfg.agent.hilp_discount)}", str(self.cfg.agent.command_injection), f"mix{str(self.cfg.agent.mix_ratio)}", str(self.cfg.use_history_action), str(self.cfg.agent.z_dim), self.cfg.load_replay_buffer.split("/")[-1], f"phh{str(self.cfg.agent.phi_hidden_dim)}", f"{str(self.cfg.agent.feature_type)}", f"sac{str(self.cfg.agent.use_sac_net)}", f"hor{str(self.cfg.agent.obs_horizon)}", f"rsz{str(self.cfg.agent.random_sample_z)}"])
        self.exp_name = exp_name
        
        cfg_dict = omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        cfg_dict['work_dir'] = self.work_dir
        if self.cfg.use_wandb:
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                        config=cfg_dict,
                        resume='allow',
                        id=wandb_run_id if not self.cfg.eval_only else None,
            )
            run_id = wandb.run.id
            with open(os.path.join(self.work_dir, "wandb.yaml"), "w") as f:
                yaml.dump({"run_id": run_id}, f)
        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        print("CFG Device: ", cfg.device)
        print("Self Device: ", self.device)
        print("Eval Env Device: ", self.eval_env.device)

        print("loading Replay from %s", self.cfg.load_replay_buffer)
        hard_coded_act_spec = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        dataset = HilbertRepreTestDataset(
            data_dir=str(cfg.load_replay_buffer),
            goal_future=float(cfg.future),
            p_randomgoal=float(cfg.p_randomgoal),
            obs_horizon=int(cfg.agent.obs_horizon),
            full_loading=True,
            use_history_action=cfg.use_history_action,
            discount=float(cfg.discount),
            load_command=self.cfg.agent.command_injection,
            dataset_ratio=float(cfg.dataset_ratio),
        )
        # train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05])
        data_loader_conf = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
        }
        self.train_dataloader = DataLoader(dataset, **data_loader_conf)
        # self.val_dataloader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=False)
        sample = dataset[0]
        print("Sample obs dim: ", sample['next_obs'].shape[-1])
        flatten_obs_dim = int(sample['next_obs'].shape[-1])
        self.flatten_obs_dim = flatten_obs_dim
        self.obs_dim = int(flatten_obs_dim // self.cfg.agent.obs_horizon)
        print("Obs dim per time step: ", self.obs_dim)
        agent_cfg = self.cfg.agent
        agent_cfg.obs_shape = (flatten_obs_dim,)
        agent_cfg.critic_obs_shape = (flatten_obs_dim,)
        self.agent = make_agent(cfg.obs_type,
                                cfg.image_wh,
                                specs.Array(shape=(self.flatten_obs_dim,), dtype=np.float32, name='obs'),
                                hard_coded_act_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                agent_cfg)
        self.example_data_buffers = {}
        self.target_traj_idx = {}
        self.raw_cosine_sim = {}
        self.raw_diff_cosine_sim = {}
        self.raw_diff_distance = {}
        self.raw_distance = {}
        
        for dirn in os.listdir(self.cfg.example_replay_path):
            full_path = os.path.join(self.cfg.example_replay_path, dirn)
            commandn = dirn.split('.')[0]
            example_data_buffer = DataBuffer.copy_from_path(full_path)
            self.example_data_buffers[commandn] = example_data_buffer
            episode_ends = example_data_buffer.episode_ends[:]
            episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
            max_traj_idx = np.argmax(episode_lengths)
            ep_start = 0 if max_traj_idx == 0 else episode_ends[max_traj_idx - 1]
            ep_end = episode_ends[max_traj_idx]
            self.target_traj_idx[commandn] = (max_traj_idx, ep_start, ep_end)
        if cfg.resume_from is not None:
            if cfg.resume_from.endswith('.pt'):
                assert os.path.exists(cfg.resume_from), f"Resume from {cfg.resume_from} does not exist"
                self.load_checkpoint(cfg.resume_from)
            else:
                models_dir = os.path.join(self.work_dir, "models")
                assert os.path.exists(models_dir), f"Models dir {models_dir} does not exist"
                print("Models Dir: ", models_dir)
                models = [filen for filen in os.listdir(models_dir) if filen.endswith('.pt')]
                models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                self._checkpoint_filepath = os.path.join(self.work_dir, "models", models[-1])
                if os.path.exists(self._checkpoint_filepath):
                    self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.resume_phi_from is not None:
            assert cfg.resume_phi_from.endswith('.pt'), f"Resume from {cfg.resume_phi_from} must be a .pt file"
            self.load_phi_from_checkpoint(cfg.resume_phi_from)

    
    def _make_eval_env(self):
        cfg = self.cfg
        # Only support legged-gym env here
        task_name = cfg.task
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=task_name)
        self.env_cfg.env.num_envs = 1
        self.env_cfg.env.episode_length_s = 1000

        # prevent in-episode command resampling; we will control commands manually
        self.env_cfg.commands.resampling_time = 1000

        # # ---- Build an argparse.Namespace for Isaac Gym / legged-gym helpers ----
        if ":" in self.cfg.device:
            compute_device_id = int(self.cfg.device.split(":")[1])
        else:
            compute_device_id = 0
        compute_device_id = compute_device_id
        graphics_device_id = compute_device_id
        headless = bool(getattr(cfg, "headless", False))

        args = build_isaac_namespace(task_name, self.env_cfg.env.num_envs, headless, compute_device_id, graphics_device_id)        
        # 地形和域随机化设置
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.noise.add_noise = True
        self.env_cfg.domain_rand.randomize_friction = True
        self.env_cfg.domain_rand.randomize_load = False
        self.env_cfg.domain_rand.randomize_gains = False
        self.env_cfg.domain_rand.randomize_link_props = False
        self.env_cfg.domain_rand.randomize_base_mass = False
        
        self.env_cfg.rewards.penalize_curriculum = False
        # 地形设置为平地
        self.env_cfg.terrain.mesh_type = 'trimesh'
        self.env_cfg.terrain.num_rows = 1
        self.env_cfg.terrain.num_cols = 1
        self.env_cfg.terrain.max_init_terrain_level = 1
        self.env_cfg.terrain.selected = True
        self.env_cfg.terrain.selected_terrain_type = "random_uniform"
        self.env_cfg.terrain.terrain_kwargs = {
            "random_uniform": {
                "min_height": -0.00,
                "max_height": 0.00,
                "step": 0.005,
                "downsampled_scale": 0.2
            },
        }
        env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=self.env_cfg)
        W, H, FPS = 720, 720, 30
        cam_props = gymapi.CameraProperties()
        cam_props.width = W
        cam_props.height = H
        self.W = W
        self.H = H
        self.FPS = FPS
        self.cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
        return env

    def _init_env_cam(self, env):
        for i in range(env.num_bodies):
            env.gym.set_rigid_body_color(
                env.envs[0], env.actor_handles[0], i,
                gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3)
            )
        self.camera_rot = np.pi * 8 / 10
        self.camera_rot_per_sec = 1 * np.pi / 10
        self.camera_relative_position = np.array([1, 0, 0.8])
        self.track_index = 0
        self.look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        _safe_set_viewer_cam(env, self.look_at + self.camera_relative_position, self.look_at, self.track_index)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def get_argmax_goal(self, custom_reward):
        num_steps = self.agent.cfg.num_inference_steps
        reward_list, next_obs_list = [], []
        batch_size = 0
        while batch_size < num_steps:
            batch = self.replay_loader.sample(self.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(self.cfg.device)
            next_obs_list.append(batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        reward, next_obs = torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
        reward_t, next_obs_t = reward[:num_steps], next_obs[:num_steps]
        return next_obs_t[torch.argmax(reward_t)].detach().cpu().numpy()

    def train(self):
        if self.cfg.eval_only:
            assert self.cfg.resume_from is not None
            if self.cfg.eval_all:
                ckpts = os.listdir(os.path.join(self.work_dir, "models"))
                ckpts = [ckpt for ckpt in ckpts if ckpt.endswith('.pt')]
                ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                for ckpt in ckpts:
                    self.load_checkpoint(os.path.join(self.work_dir, "models", ckpt))
                    self.eval()
            else:
                self.eval()
            return
        while True:
            metrics_summon = defaultdict(list)
            # eval_num_batches = self.cfg.eval_every_steps // 20
            for batch in tqdm(self.train_dataloader):
                if self.global_step % self.cfg.eval_every_steps == 0:
                    _checkpoint_filepath = os.path.join(self.work_dir, "models", f"{self.global_step}.pt")
                    self.save_checkpoint(_checkpoint_filepath)
                    self.remove_outdated_cktps()
                    self.eval() 
                    # self.eval_data(num_batches=eval_num_batches)
                metrics = self.agent.update_batch(batch, self.global_step)
                for k, v in metrics.items():
                    metrics_summon[k].append(v)
                if (self.global_step + 1) % self.cfg.log_every_steps == 0:
                    for k, v in metrics_summon.items():
                        metrics_summon[k] = np.mean(v)
                    if self.cfg.use_wandb:
                        wandb.log({f"train/{'_'.join(k.split('/'))}" if "/" in k else f"train/{k}": v for k, v in metrics_summon.items()}, step=self.global_step)
                    else:
                        for k, v in metrics_summon.items():
                            print(f"train/{'_'.join(k.split('/'))}" if "/" in k else f"train/{k}: {v}")
                    metrics_summon = defaultdict(list)
                self.global_step += 1           
                if self.global_step >= (self.cfg.num_grad_steps // self.cfg.action_repeat):
                    break
            if self.global_step >= (self.cfg.num_grad_steps // self.cfg.action_repeat):
                break
        _checkpoint_filepath = os.path.join(self.work_dir, "models", f"{self.global_step}.pt")
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint

    # @torch.no_grad()
    # def eval_data(self, num_batches=100):
    #     self.agent.feature_learner.eval()
    #     summon_metrics = defaultdict(list)
    #     for idx, batch in tqdm(enumerate(self.val_dataloader), total=num_batches):
    #         if idx >= num_batches:
    #             break
    #         metrics = self.agent.update_batch(batch, self.global_step, is_train=False)
    #         for k, v in metrics.items():
    #             summon_metrics[k].append(v)
    #     for k, v in summon_metrics.items():
    #         summon_metrics[k] = np.mean(v)
    #     if self.cfg.use_wandb:
    #         wandb.log({f"eval/{k}": v for k, v in summon_metrics.items()}, step=self.global_step)
    #     else:
    #         for k, v in summon_metrics.items():
    #             print(f"eval/{k}: {v}")
    #     self.agent.feature_learner.train()
            
    def _proprocess_obs(self, obs):
        if self.cfg.agent.obs_horizon > 1:
            pure_obs = obs[..., -self.cfg.agent.obs_horizon:, :self.obs_dim].reshape(1, -1)
        else:
            pure_obs = obs[..., -1, :self.obs_dim].reshape(1, -1)
        if self.cfg.use_history_action:
            obs_command = obs[:, -1, self.obs_dim:self.obs_dim + 11].reshape(1, -1)
            assert self.obs_dim + 13 == obs.shape[-1], f"obs_dim + 32 != obs.shape[-1], raw_obs_dim: {obs.shape[-1]} obs_dim: {self.obs_dim} sum: {self.obs_dim + 13} != {obs.shape[-1]}"
        else:
            obs_command = obs[:, -1, self.obs_dim + 19:self.obs_dim + 30].reshape(1, -1)
            assert self.obs_dim + 32 == obs.shape[-1], f"obs_dim + 13 != obs.shape[-1], raw_obs_dim: {obs.shape[-1]} obs_dim: {self.obs_dim} sum: {self.obs_dim + 32} != {obs.shape[-1]}"
        return pure_obs, obs_command

    def plot_traj_images(self, phi_list, image_parent, command_name, img_internal_id, z_actor=None):
        phi_traj = torch.cat(phi_list, axis=0)
        hilbert_traj = phi_traj.cpu().numpy()
        # z_diff = torch.diff(hilbert_traj, dim=0)
        # z_diff_normed = math.sqrt(self.cfg.agent.z_dim) * F.normalize(z_diff, dim=1)
        # z_diff_normed = z_diff_normed.cpu().numpy()
        goal_z_cosine_sim_list, goal_distance_list, goal_absdist_list = calc_z_vector_matrixes(hilbert_traj, goal_vector=hilbert_traj[-1])
        # plot_per_step_z(
        #     latent=z_diff_normed,
        #     eid=img_internal_id,
        #     command_name=command_name,
        #     out_dir=image_parent,   
        # )
        middle_goal_z_cosine_sim_list, middle_goal_distance_list, middle_goal_absdist_list = calc_z_vector_matrixes(hilbert_traj, goal_vector=hilbert_traj[hilbert_traj.shape[0] // 2])
        plot_tripanel_heatmaps_with_line(
            goal_z_cosine_sim_list,
            goal_distance_list,
            goal_absdist_list,
            [f"goal={goal_z_cosine_sim_list[g_idx].shape[-1]}_{command_name}_{img_internal_id}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
            image_parent,
            title_cos=f'{command_name} {img_internal_id} Z Cosine Similarity',
            title_dist=f'{command_name} {img_internal_id} Latent Space Distance',
        )
        plot_tripanel_heatmaps_with_line(
            middle_goal_z_cosine_sim_list,
            middle_goal_distance_list,
            middle_goal_absdist_list,
            [f"middle_goal={middle_goal_z_cosine_sim_list[g_idx].shape[-1]}_{command_name}_{img_internal_id}.png" for g_idx in range(len(middle_goal_z_cosine_sim_list))],
            image_parent,
            title_cos=f'{command_name} {img_internal_id} Z Cosine Similarity',
            title_dist=f'{command_name} {img_internal_id} Latent Space Distance',
        )
        if z_actor is not None:
            goal_z_cosine_sim_list, goal_distance_list, goal_absdist_list = calc_z_vector_matrixes(hilbert_traj, goal_vector=z_actor.cpu().numpy())
            plot_tripanel_heatmaps_with_line(
                goal_z_cosine_sim_list,
                goal_distance_list,
                goal_absdist_list,
                [f"goal=z_actor_{command_name}_{img_internal_id}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
                image_parent,
                title_cos=f'z_actor_{command_name} {img_internal_id} Z Cosine Similarity',
                title_dist=f'z_actor_{command_name} {img_internal_id} Latent Space Distance',
            )
 
    def get_traj_data_from_example(self, data_buffer, key):
        command_vec = torch.tensor(data_buffer.meta['episode_command'][0], device=self.eval_env.device)
        traj_idx, ep_start, ep_end = self.target_traj_idx[key]
        ep_len = ep_end - ep_start
        raw_state_data = data_buffer.data['proprio'][ep_start:ep_end]
        ep_start_obs = data_buffer.meta['ep_start_obs'][traj_idx]
        raw_state = np.concatenate([ep_start_obs[..., :raw_state_data.shape[-1]], raw_state_data], axis=0)
        raw_index = np.arange(raw_state.shape[0] - self.cfg.agent.obs_horizon + 1)
        horizon_index = raw_index[:, None] + np.arange(self.cfg.agent.obs_horizon)
        raw_state = raw_state[horizon_index]
        traj = raw_state.reshape(raw_state.shape[0], -1)
        traj = traj[-ep_len:]
        return traj, command_vec, traj_idx

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode')

    def save_checkpoint(self, fp: tp.Union[Path, str]) -> None:
        """保存关键状态用于断点重训。

        保存内容包含：agent、global_step、global_episode、replay_loader（可被 only/exclude 调整）。
        """
        # logger.info(f"Saving checkpoint to {fp}")
        print(f"Saving checkpoint to {fp}")
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def remove_outdated_cktps(self):
        models_dir = os.path.join(self.work_dir, "models")
        models = os.listdir(models_dir)
        models.sort(key=lambda x: int(x.split('.')[0]))
        for model in models[:-20]:
            os.remove(os.path.join(models_dir, model))

    def load_checkpoint(self, fp: tp.Union[Path, str], use_pixels=False) -> None:
        """从磁盘加载 checkpoint。

        - only：仅恢复指定键（例如只加载 replay_loader）。
        - use_pixels：当使用像素观测时，将存储中的 'pixel' 字段重命名为 'observation'。
        """
        print(f"Loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f, map_location='cpu')
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')
        for name, val in payload.items():
            # logger.info("Reloading %s from %s", name, fp)
            print(f"Reloading {name} from {fp}")
            if name == "agent":
                self.agent.init_from(val)
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    print(f"Reloaded agent at global episode {self.global_episode}")
                    # logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def load_phi_from_checkpoint(self, fp: tp.Union[Path, str]) -> None:
        print(f"Loading phi from checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f, map_location='cpu')
        feature_learner_state_dict = payload['agent'].feature_learner.state_dict()
        self.agent.feature_learner.load_state_dict(feature_learner_state_dict)


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
