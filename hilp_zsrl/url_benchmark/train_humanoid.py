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
from url_benchmark.hilbert_dataset import HilbertRepresentationDatasetLegacy, HilbertRepresentationDataset
from isaacgym import gymapi
from url_benchmark.legged_gym_env_utils import build_isaac_namespace, _to_rgb_frame
from torch.utils.data import DataLoader
from url_benchmark.dataset_utils import InfiniteDataLoaderWrapper
from collections import defaultdict
# pbar import
import datetime
from tqdm import tqdm
from url_benchmark.humanoid_utils import _safe_set_viewer_cam, get_cosine_sim, collect_nonzero_losses, calc_phi_loss_upper, calc_z_vector_matrixes, plot_matrix_heatmaps, compute_global_color_limits, plot_tripanel_heatmaps_with_line, plot_per_step_z


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
    eval_every_steps: int = 10000
    num_final_eval_episodes: int = 50
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    # training
    num_grad_steps: int = 100000000
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

        # self.train_env = self._make_env()  # 环境仅用于读取规格与评估
        self.eval_env = self._make_eval_env()
        self._init_env_cam(self.eval_env)
        print("CFG Device: ", cfg.device)
        print("Self Device: ", self.device)
        print("Eval Env Device: ", self.eval_env.device)

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
        hard_coded_act_spec = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        dataset = HilbertRepresentationDataset(
            data_dir=str(cfg.load_replay_buffer),
            goal_future=float(cfg.future),
            p_randomgoal=float(cfg.p_randomgoal),
            obs_horizon=int(cfg.agent.obs_horizon),
            full_loading=True,
            use_history_action=cfg.use_history_action,
            discount=float(cfg.discount),
            load_command=self.cfg.agent.command_injection,
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05])
        data_loader_conf = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
        }
        self.train_dataloader = DataLoader(train_set, **data_loader_conf)
        self.val_dataloader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=False)
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
            episode_lengths = np.diff(episode_ends)
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
            eval_num_batches = self.cfg.eval_every_steps // 20
            for batch in tqdm(self.train_dataloader):
                if self.global_step % self.cfg.eval_every_steps == 0:
                    _checkpoint_filepath = os.path.join(self.work_dir, "models", f"{self.global_step}.pt")
                    self.save_checkpoint(_checkpoint_filepath)
                    self.eval() 
                    self.eval_data(num_batches=eval_num_batches)
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

    @torch.no_grad()
    def eval_data(self, num_batches=100):
        self.agent.feature_learner.eval()
        summon_metrics = defaultdict(list)
        for idx, batch in tqdm(enumerate(self.val_dataloader), total=num_batches):
            if idx >= num_batches:
                break
            metrics = self.agent.update_batch(batch, self.global_step, is_train=False)
            for k, v in metrics.items():
                summon_metrics[k].append(v)
        for k, v in summon_metrics.items():
            summon_metrics[k] = np.mean(v)
        if self.cfg.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in summon_metrics.items()}, step=self.global_step)
        else:
            for k, v in summon_metrics.items():
                print(f"eval/{k}: {v}")
        self.agent.feature_learner.train()
            
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
        plot_tripanel_heatmaps_with_line(
            goal_z_cosine_sim_list,
            goal_distance_list,
            goal_absdist_list,
            [f"goal={goal_z_cosine_sim_list[g_idx].shape[-1]}_{img_internal_id}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
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

    def _env_rollout(self, 
        command_vec,
        command_name,
        video_path,
        rewards_json_path,
        eval_steps,
        goal_type,
        image_parent,
        command_horizon=None,
        goals_list=None,
        z_actor=None,
    ):
        _, _ = self.eval_env.reset()
        self.eval_env.commands[:, :10] = command_vec
        obs, critic_obs, _, _, _ = self.eval_env.step(torch.zeros(
            self.eval_env.num_envs, self.eval_env.num_actions, dtype=torch.float, device=self.eval_env.device))
        if self.cfg.save_video:
            look_at = np.array(self.eval_env.root_states[0, :3].cpu(), dtype=np.float64)
            _safe_set_viewer_cam(self.eval_env, look_at + self.camera_relative_position, look_at, self.track_index)
            # ==================== 相机传感器与视频写出器 ====================
            frame_skip = max(1, int(round(1.0 / (self.FPS * self.eval_env.dt))))  # 模拟 -> 视频帧率下采样
            # 让传感器相机与 viewer 相机初始对齐
            self.eval_env.gym.set_camera_location(
                self.cam_handle, self.eval_env.envs[0],
                gymapi.Vec3(*(look_at + self.camera_relative_position)),
                gymapi.Vec3(*look_at)
            )
            # imageio 写 mp4，依赖 ffmpeg（imageio-ffmpeg）
            writer = imageio.get_writer(video_path, fps=self.FPS)
        timestep = 0
        pure_obs, obs_command = self._proprocess_obs(obs)
        last_index = -1
        dones = np.zeros(self.eval_env.num_envs, dtype=np.bool_)
        rewards_recorder = {}
        rewards_recorder['env_rewards'] = 0
        rewards_recorder['zdiff_rewards'] = 0
        rewards_recorder['env_rew_list'] = []
        rewards_recorder['zdiff_rew_list'] = []
        rewards_recorder['zstate_rewards'] = 0
        rewards_recorder['zstate_rew_list'] = []
        phi_list = []
        full_traj_phi = []
        img_internal_id = 0
        while not dones.any() and timestep < eval_steps:
            with torch.inference_mode():
                if goal_type == 'raw_cmd':
                    assert self.agent.command_injection or self.agent.use_raw_command, f"command_injection or use_raw_command must be True"
                    z_hilbert, z_actor = self.agent.sample_z(1, obs_command)
                elif "horizon" in goal_type:
                    assert command_horizon is not None and goals_list is not None, "command_horizon and goals_list must be provided"
                    assert "state" in goal_type or "diff" in goal_type, "goal_type must be state_horizon or diff_horizon"
                    if timestep // command_horizon != last_index and last_index < len(goals_list) - 1:
                        print(f"Switch goal state: {last_index} -> {timestep // command_horizon} / {timestep} / {eval_steps}")
                        last_index = timestep // command_horizon
                        last_index = min(last_index, len(goals_list) - 1)
                        if len(phi_list) > 0:
                            self.plot_traj_images(phi_list, image_parent, command_name, img_internal_id, z_actor=z_actor)
                            phi_list = []
                    meta = self.agent.get_goal_meta(goal_array=goals_list[last_index].squeeze(0), obs_array=None if "state" in goal_type else pure_obs.squeeze(0))
                    z_actor = torch.tensor(meta['z'], device=self.eval_env.device).reshape(1, -1)       
                    z_hilbert = z_actor
                    img_internal_id += 1
                elif "fit" in goal_type:
                    assert z_actor is not None, "z_actor must be provided"    
                    z_hilbert = z_actor
                # print(pure_obs.shape, z_actor.shape)
                actions, _ = self.agent.actor.act_inference(pure_obs, z_actor)
                phi = self.agent.feature_learner.feature_net(pure_obs)
                phi_list.append(phi)
                full_traj_phi.append(phi)
            last_obs = pure_obs
            obs, critic_obs, reward, dones, _ = self.eval_env.step(actions)
            pure_obs, obs_command = self._proprocess_obs(obs)
            self.eval_env.commands[:, :10] = command_vec
            with torch.inference_mode():
                zdiff_reward, zstate_reward = self.agent.get_z_rewards(last_obs, pure_obs, z_hilbert.cpu().numpy())
            rewards_recorder['env_rew_list'].append(float(reward.item()))
            rewards_recorder['zdiff_rew_list'].append(float(zdiff_reward))
            rewards_recorder['zstate_rew_list'].append(float(zstate_reward))
            # print(f"z_reward: {z_reward}, env_reward: {reward.item()}")
            if dones.any():
                print(f"goal: {goal_type}, command: {command_name}, env reward: {np.sum(rewards_recorder['env_rew_list'])}, zdiff reward: {np.sum(rewards_recorder['zdiff_rew_list'])}, zstate reward: {np.sum(rewards_recorder['zstate_rew_list'])}, ep_len: {len(rewards_recorder['env_rew_list'])}")
            if self.cfg.save_video:
                # ===== 相机跟踪与旋转 =====
                look_at = np.array(self.eval_env.root_states[0, :3].cpu(), dtype=np.float64)
                camera_rot = (self.camera_rot + self.camera_rot_per_sec * self.eval_env.dt) % (2 * np.pi)
                h_scale, v_scale = 1.0, 0.8
                camera_relative_position = 2 * np.array(
                    [np.cos(camera_rot) * h_scale, np.sin(camera_rot) * h_scale, 0.5 * v_scale]
                )
                # 更新 viewer 相机
                _safe_set_viewer_cam(self.eval_env, look_at + camera_relative_position, look_at, self.track_index)
                # env.set_camera(look_at + camera_relative_position, look_at, track_index)
                # 同步传感器相机（用于录制）
                self.eval_env.gym.set_camera_location(
                    self.cam_handle, self.eval_env.envs[0],
                    gymapi.Vec3(*(look_at + camera_relative_position)),
                    gymapi.Vec3(*look_at)
                )
                # ===== 抓帧与写视频（CPU 路径）=====
                if timestep % frame_skip == 0:
                    # 确保渲染管线推进
                    self.eval_env.gym.step_graphics(self.eval_env.sim)
                    self.eval_env.gym.render_all_camera_sensors(self.eval_env.sim)

                    img_any = self.eval_env.gym.get_camera_image(
                        self.eval_env.sim, self.eval_env.envs[0], self.cam_handle, gymapi.IMAGE_COLOR
                    )
                    rgb = _to_rgb_frame(img_any, self.H, self.W)
                    writer.append_data(rgb)
            # ===== 干扰和中断设置 =====
            self.eval_env.use_disturb = True
            self.eval_env.disturb_masks[:] = True
            self.eval_env.disturb_isnoise[:] = True
            self.eval_env.disturb_rad_curriculum[:] = 1.0
            self.eval_env.interrupt_mask[:] = self.eval_env.disturb_masks[:]
            self.eval_env.standing_envs_mask[:] = True
            timestep += 1
        if len(phi_list) > 0:
            self.plot_traj_images(phi_list, image_parent, command_name, img_internal_id, z_actor=z_actor)
        if "horizon" in goal_type:
            self.plot_traj_images(full_traj_phi, image_parent, command_name, img_internal_id)
        if self.cfg.save_video:
            writer.close()
            print("Saved video to %s", video_path)
        tmp_ = {}
        for k, v in rewards_recorder.items():
            tmp_[k.replace("list", "sum")] = np.sum(v)
            tmp_[k.replace("list", "mean")] = np.mean(v)
        rewards_recorder.update(tmp_)
        with open(rewards_json_path, 'w') as f:
            json.dump(rewards_recorder, f, indent=4)
        ep_len = len(rewards_recorder['env_rew_list'])
        rewards_recorder = tmp_
        if goal_type == "raw_cmd":
            command_name = command_name + "_raw_cmd"
        elif "horizon" in goal_type:
            if "state" in goal_type:
                command_name = command_name + f"_stateg{command_horizon}"
            elif "diff" in goal_type:
                command_name = command_name + f"_diffg{command_horizon}"
            else:
                raise ValueError(f"Invalid goal type: {goal_type}")
        elif "fit" in goal_type:
            command_name = command_name + f"_{goal_type}"
        metrics = {f"{command_name}_ep_len": ep_len}
        for k, v in rewards_recorder.items():
            metrics[f"{command_name}_{k}"] = v
        if self.cfg.use_wandb:
            wandb.log({f"eval_detail/{k}": v for k, v in metrics.items()}, step=self.global_step)
        else:
            for k, v in metrics.items():
                print(f"eval_detail/{k}: {v}")
        return {k: v for k, v in rewards_recorder.items()}
    
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

    def eval(self):
        self.agent.feature_learner.eval()
        commands_horizons = [10, 20, 40, 80]
        eval_time = 10
        if self.cfg.eval_only:
            eval_parent = self.work_dir.replace("exp_local", "eval_only")
            eval_save_parent = os.path.join(eval_parent, f"{self.global_step}")
        else:
            eval_save_parent = os.path.join(self.work_dir, "eval_result", f"{self.global_step}")
        print("Eval Save Parent: ", eval_save_parent)
        shutil.rmtree(eval_save_parent, ignore_errors=True)
        os.makedirs(eval_save_parent, exist_ok=True)
        eval_steps = int(eval_time / self.eval_env.dt)
        # cos_sim_save_parent = os.path.join(eval_save_parent, "images", "cos_sims")
        # phi_loss_save_parent = os.path.join(eval_save_parent, "images", "phi_losses")
        goal_cos_sim_save_parent = os.path.join(eval_save_parent, "images", "goal_cos_sims")
        rollout_parent = os.path.join(eval_save_parent, "rollout")
        shutil.rmtree(rollout_parent, ignore_errors=True)
        # shutil.rmtree(cos_sim_save_parent, ignore_errors=True)
        # shutil.rmtree(phi_loss_save_parent, ignore_errors=True)
        shutil.rmtree(goal_cos_sim_save_parent, ignore_errors=True)
        os.makedirs(rollout_parent, exist_ok=True)
        # os.makedirs(cos_sim_save_parent, exist_ok=True)
        # os.makedirs(phi_loss_save_parent, exist_ok=True)
        os.makedirs(goal_cos_sim_save_parent, exist_ok=True)
        # # 全局收集
        # phi_losses = []
        # phi_loss_mats = []        # 各 episode 的 φ-loss 矩阵
        # phi_loss_mats_names = []      
        traj_data = {}
        for key, data_buffer in self.example_data_buffers.items():
            traj_data[key] = self.get_traj_data_from_example(data_buffer, key)
            
        for key, (traj, command_vec, traj_idx) in traj_data.items():
            current_goal_path = os.path.join(goal_cos_sim_save_parent, f"{key}")
            os.makedirs(current_goal_path, exist_ok=True)
            command_name = key
            
            z, hilbert_traj = self.agent.get_traj_meta(traj)  # (traj_len, z_dim)
            goal_z_cosine_sim_list, goal_distance_list, goal_absdist_list = calc_z_vector_matrixes(hilbert_traj)

            # # draw a hot map of z, with metric as the cosine similarity between all z of different time steps
            # if command_name not in self.raw_cosine_sim:
            #     self.raw_cosine_sim[command_name] = get_cosine_sim(traj)
            #     raw_diff = np.diff(traj, axis=0)
            #     self.raw_diff_cosine_sim[command_name] = get_cosine_sim(raw_diff)
            #     self.raw_diff_distance[command_name] = np.linalg.norm(raw_diff, axis=-1)
            #     self.raw_distance[command_name] = np.linalg.norm(traj, axis=-1)
            # raw_cos_sim = self.raw_cosine_sim[command_name]
            # raw_diff_cos_sim = self.raw_diff_cosine_sim[command_name]
            # raw_diff_distance = self.raw_diff_distance[command_name]
            # raw_distance = self.raw_distance[command_name]
            # plot_per_step_z(
            #     latent=z,
            #     eid=traj_idx,
            #     out_dir=cos_sim_save_parent,
            #     command_name=command_name,
            #     cos_raw=raw_cos_sim,
            #     cos_raw_diff=raw_diff_cos_sim,
            #     raw_diff_distance=raw_diff_distance,
            #     raw_distance=raw_distance,   
            # )
            # phi_loss_matrix = calc_phi_loss_upper(hilbert_traj, gamma=self.cfg.agent.hilp_discount)
            # phi_loss = collect_nonzero_losses(phi_loss_matrix)
            # phi_losses.extend(phi_loss)
            # phi_loss_mats.append(phi_loss_matrix)
            # phi_loss_mats_names.append(f"phi_loss_latent_{command_name}_{traj_idx}.png")
            plot_tripanel_heatmaps_with_line(
                goal_z_cosine_sim_list,
                goal_distance_list,
                goal_absdist_list,
                [f"goal={goal_z_cosine_sim_list[g_idx].shape[-1]}_{traj_idx}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
                current_goal_path,
                title_cos=f'{command_name}_{traj_idx} Z Cosine Similarity',
                title_dist=f'{command_name}_{traj_idx} Latent Space Distance',
            )
        # vmin, vmax = compute_global_color_limits(phi_loss_mats)
        # plot_matrix_heatmaps(phi_loss_mats, phi_loss_mats_names, phi_loss_save_parent, vmin, vmax)
        
        if self.cfg.agent.command_injection or self.cfg.agent.use_raw_command:
            eval_results = defaultdict(list)
            for key, (traj, command_vec, traj_idx) in traj_data.items():
                command_name = key  
                video_path = os.path.join(rollout_parent, "cmd", f"{command_name}_raw_cmd.mp4")
                rewards_json_path = os.path.join(rollout_parent, "cmd", f"{command_name}_raw_cmd.json")
                image_parent = os.path.join(rollout_parent, "cmd", "images")
                os.makedirs(image_parent, exist_ok=True)

                rollout_results = self._env_rollout(
                    command_name=command_name, 
                    goal_type="raw_cmd",
                    command_vec=command_vec, 
                    video_path=video_path,
                    rewards_json_path=rewards_json_path, 
                    eval_steps=eval_steps,
                    image_parent=image_parent,
                )
                for k, v in rollout_results.items():
                    eval_results[k].append(v) 
            eval_results = {key: np.mean(eval_results[key]) for key in eval_results if len(eval_results[key]) > 0}
            if self.cfg.use_wandb:
                wandb.log({f"eval_mean/{k}_raw_cmd": v for k, v in eval_results.items()}, step=self.global_step)
            else:
                for k, v in eval_results.items():
                    print(f"eval_mean/{k}_raw_cmd: {v}")
        else:
            eval_fit_state = defaultdict(list)
            eval_fit_diff = defaultdict(list)
            eval_fit_diag_state = defaultdict(list)
            eval_fit_diag_diff = defaultdict(list)
            for key, data_buffer in self.example_data_buffers.items():
                command_name = key
                env_commands = data_buffer.meta['episode_command'][0]
                command_vec = torch.tensor(env_commands, device=self.eval_env.device)
                episode_ends = data_buffer.episode_ends[:]
                obs_t = []
                next_obs_t = []
                reward_t = []
                for ep_id in range(len(episode_ends)):
                    ep_start = 0 if ep_id == 0 else episode_ends[ep_id - 1]
                    ep_end = episode_ends[ep_id]
                    ep_len = ep_end - ep_start
                    ep_start_obs = data_buffer.meta['ep_start_obs'][ep_id]
                    raw_state_data = data_buffer.data['proprio'][ep_start:ep_end]
                    raw_state = np.concatenate([ep_start_obs[..., :raw_state_data.shape[-1]], raw_state_data], axis=0)
                    raw_index = np.arange(raw_state.shape[0] - self.cfg.agent.obs_horizon + 1)
                    horizon_index = raw_index[:, None] + np.arange(self.cfg.agent.obs_horizon)
                    raw_state = raw_state[horizon_index]
                    traj = raw_state.reshape(raw_state.shape[0], -1)
                    traj = traj[-ep_len:]
                    obs_array = traj[:-1]
                    next_obs_array = traj[1:]

                    reward_array = data_buffer.data['rewards'][ep_start:ep_end - 1]
                    obs_t.append(torch.as_tensor(obs_array))
                    next_obs_t.append(torch.as_tensor(next_obs_array))
                    reward_t.append(torch.as_tensor(reward_array))
                obs_t = torch.cat(obs_t, 0).to(self.eval_env.device)
                obs_t = obs_t.view(obs_t.shape[0], -1)
                next_obs_t = torch.cat(next_obs_t, 0).to(self.eval_env.device)
                next_obs_t = next_obs_t.view(next_obs_t.shape[0], -1)
                reward_t = torch.cat(reward_t, 0).to(self.eval_env.device)
                # print("Obs T Device: ", obs_t.device, "Obs T shape:", obs_t.shape)
                meta_state, diag_state = self.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t, feat_type='state')
                meta_diff, diag_diff = self.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t, feat_type='diff')

                z_actor_state = torch.tensor(meta_state['z'], device=self.eval_env.device).reshape(1, -1)
                z_actor_diff = torch.tensor(meta_diff['z'], device=self.eval_env.device).reshape(1, -1)

                fit_video_path_state = os.path.join(rollout_parent, f"fit_state_{command_name}.mp4")
                fit_rewards_json_path_state = os.path.join(rollout_parent, f"fit_state_{command_name}.json")
                fit_video_path_diff = os.path.join(rollout_parent, f"fit_diff_{command_name}.mp4")
                fit_rewards_json_path_diff = os.path.join(rollout_parent, f"fit_diff_{command_name}.json")
                image_state_parent = os.path.join(rollout_parent, "fit_state_images")
                image_diff_parent = os.path.join(rollout_parent, "fit_diff_images")
                os.makedirs(image_state_parent, exist_ok=True)
                os.makedirs(image_diff_parent, exist_ok=True)
                rollout_results = self._env_rollout(
                    command_name=command_name, 
                    goal_type="fit_state",
                    command_vec=command_vec, 
                    video_path=fit_video_path_state,
                    rewards_json_path=fit_rewards_json_path_state, 
                    eval_steps=eval_steps,
                    z_actor=z_actor_state,
                    image_parent=image_state_parent,
                )
                for k, v in rollout_results.items():
                    eval_fit_state[k].append(v)
                for k, v in diag_state.items():
                    eval_fit_diag_state[k].append(v)
                if self.cfg.use_wandb:
                    wandb.log({f"eval_fit_diag_state_detail/{command_name}_{k}": v for k, v in diag_state.items()}, step=self.global_step)
                else:
                    for k, v in diag_state.items():
                        print(f"eval_fit_diag_state_detail/{command_name}_{k}: {v}")
                rollout_results = self._env_rollout(
                    command_name=command_name, 
                    goal_type="fit_diff",
                    command_vec=command_vec, 
                    video_path=fit_video_path_diff,
                    rewards_json_path=fit_rewards_json_path_diff, 
                    eval_steps=eval_steps,
                    z_actor=z_actor_diff,
                    image_parent=image_diff_parent,
                )
                for k, v in rollout_results.items():
                    eval_fit_diff[k].append(v)
                for k, v in diag_diff.items():
                    eval_fit_diag_diff[k].append(v)
                if self.cfg.use_wandb:
                    wandb.log({f"eval_fit_diag_diff_detail/{command_name}_{k}": v for k, v in diag_diff.items()}, step=self.global_step)
                else:
                    for k, v in diag_diff.items():
                        print(f"eval_fit_diag_diff_detail/{command_name}_{k}: {v}")
            eval_fit_state = {key: np.mean(eval_fit_state[key]) for key in eval_fit_state.keys()}
            eval_fit_diff = {key: np.mean(eval_fit_diff[key]) for key in eval_fit_diff.keys()}
            eval_fit_diag_state = {key: np.mean(eval_fit_diag_state[key]) for key in eval_fit_diag_state.keys()}
            eval_fit_diag_diff = {key: np.mean(eval_fit_diag_diff[key]) for key in eval_fit_diag_diff.keys()}
            if self.cfg.use_wandb:
                wandb.log({f"eval_fit/state_{k}": v for k, v in eval_fit_state.items()}, step=self.global_step)
                wandb.log({f"eval_fit/diff_{k}": v for k, v in eval_fit_diff.items()}, step=self.global_step)
                wandb.log({f"eval_fit/state_diag_{k}": v for k, v in eval_fit_diag_state.items()}, step=self.global_step)
                wandb.log({f"eval_fit/diff_diag_{k}": v for k, v in eval_fit_diag_diff.items()}, step=self.global_step)
            else:
                for k, v in eval_fit_state.items():
                    print(f"eval_fit/state_{k}: {v}")
                for k, v in eval_fit_diff.items():
                    print(f"eval_fit/diff_{k}: {v}")
                for k, v in eval_fit_diag_state.items():
                    print(f"eval_fit/state_diag_{k}: {v}")
                for k, v in eval_fit_diag_diff.items():
                    print(f"eval_fit/diff_diag_{k}: {v}")

            eval_diff = defaultdict(list)
            eval_state = defaultdict(list)
            for command_horizon in commands_horizons:
                diff_horizon_results = defaultdict(list)
                state_horizon_results = defaultdict(list)
                for key, (traj, command_vec, traj_idx) in traj_data.items():
                    command_name = key
                    diff_video_path = os.path.join(rollout_parent, f"diff_{command_name}_{command_horizon}.mp4")
                    diff_rewards_json_path = os.path.join(rollout_parent, f"diff_{command_name}_{command_horizon}.json")
                    state_video_path = os.path.join(rollout_parent, f"state_{command_name}_{command_horizon}.mp4")
                    state_rewards_json_path = os.path.join(rollout_parent, f"state_{command_name}_{command_horizon}.json")

                    len_trajectory = traj.shape[0]
                    resample_steps_list = np.array([i for i in range(command_horizon, len_trajectory, command_horizon)])

                    goals_list = [traj[i].reshape(1, -1) for i in resample_steps_list]
                    goals_list.append(traj[-1].reshape(1, -1))  

                    image_diff_parent = os.path.join(rollout_parent, f"{command_horizon}_diff_images")
                    image_state_parent = os.path.join(rollout_parent, f"{command_horizon}_state_images")
                    os.makedirs(image_diff_parent, exist_ok=True)
                    os.makedirs(image_state_parent, exist_ok=True)
                    rollout_results = self._env_rollout(
                        command_name=command_name, 
                        goal_type="diff_horizon",
                        command_vec=command_vec, 
                        video_path=diff_video_path,
                        rewards_json_path=diff_rewards_json_path, 
                        eval_steps=eval_steps,
                        command_horizon=command_horizon,
                        goals_list=goals_list,
                        image_parent=image_diff_parent,
                    )
                    for k, v in rollout_results.items():
                        diff_horizon_results[k].append(v)
                    rollout_results = self._env_rollout(
                        command_name=command_name, 
                        goal_type="state_horizon",
                        command_vec=command_vec, 
                        video_path=state_video_path,
                        rewards_json_path=state_rewards_json_path, 
                        eval_steps=eval_steps,
                        command_horizon=command_horizon,
                        goals_list=goals_list,
                        image_parent=image_state_parent,
                    )
                    for k, v in rollout_results.items():
                        state_horizon_results[k].append(v)
                diff_horizon_metrics = {}
                state_horizon_metrics = {}
                for k, v in diff_horizon_results.items():
                    diff_horizon_metrics[f"{command_horizon}_{k}"] = np.mean(v)
                    eval_diff[k].append(np.mean(v))
                for k, v in state_horizon_results.items():
                    state_horizon_metrics[f"{command_horizon}_{k}"] = np.mean(v)
                    eval_state[k].append(np.mean(v))
                if self.cfg.use_wandb:
                    wandb.log({f"eval_diff_horizon/{k}": v for k, v in diff_horizon_metrics.items()}, step=self.global_step)
                    wandb.log({f"eval_state_horizon/{k}": v for k, v in state_horizon_metrics.items()}, step=self.global_step)
                else:
                    for k, v in diff_horizon_metrics.items():
                        print(f"eval_diff_horizon/{k}: {v}")
                    for k, v in state_horizon_metrics.items():
                        print(f"eval_state_horizon/{k}: {v}")
            eval_diff = {key: np.mean(eval_diff[key]) for key in eval_diff.keys()}
            eval_state = {key: np.mean(eval_state[key]) for key in eval_state.keys()}
            if self.cfg.use_wandb:
                wandb.log({f"eval_diff/{k}": v for k, v in eval_diff.items()}, step=self.global_step)
                wandb.log({f"eval_state/{k}": v for k, v in eval_state.items()}, step=self.global_step)
            else:
                for k, v in eval_diff.items():
                    print(f"eval_diff/{k}: {v}")
                for k, v in eval_state.items():
                    print(f"eval_state/{k}: {v}")
        self.agent.feature_learner.train()

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
