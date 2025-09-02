import os
from pathlib import Path
import typing as tp
import dataclasses
import tempfile

import hydra
from hydra.core.config_store import ConfigStore
import omegaconf as omgcf
import torch
import numpy as np
from dm_env import specs
from torch.utils.data import DataLoader

from url_benchmark.logger import Logger
from url_benchmark import utils
from url_benchmark.hilbert_dataset import HilbertRepresentationDataset
from url_benchmark.agent import sf_dl as agents


@dataclasses.dataclass
class Config:
    agent: tp.Any
    run_group: str = "Debug"
    seed: int = 1
    device: str = "cuda"
    use_tb: bool = False
    use_wandb: bool = True
    experiment: str = "offline"

    # task/dataset
    task: str = "h1int"
    obs_type: str = "states"
    image_wh: int = 64
    action_repeat: int = 1
    discount: float = 0.98
    future: float = 0.99
    p_currgoal: float = 0.0
    p_randomgoal: float = 0.5
    hilbert_obs_horizon: int = 5
    hilbert_types: tp.Optional[tp.List[str]] = None
    hilbert_max_episodes_per_type: tp.Optional[int] = None
    use_history_action: bool = True

    # training
    num_grad_steps: int = 100000
    log_every_steps: int = 1000
    eval_every_steps: int = 0
    batch_size: int = omgcf.II("agent.batch_size")

    # dataset path
    load_replay_buffer: tp.Optional[str] = None


ConfigStore.instance().store(name="workspace_config_dl", node=Config)


def make_agent(obs_dim: int, action_dim: int, cfg: omgcf.DictConfig) -> agents.SFAgent:
    cfg.obs_type = cfg.obs_type
    cfg.image_wh = cfg.image_wh
    cfg.obs_shape = (obs_dim,)
    cfg.action_shape = (action_dim,)
    cfg.num_expl_steps = 0
    return hydra.utils.instantiate(cfg)


def legged_collate(cfg: Config):
    def _collate(batch: tp.List[tp.Dict[str, torch.Tensor]]) -> tp.Dict[str, torch.Tensor]:
        out: tp.Dict[str, torch.Tensor] = {}
        # stack required keys
        keys = batch[0].keys()
        for k in keys:
            if k == 'goal_obs':
                continue
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        # map goal_obs -> future_obs
        out['future_obs'] = torch.stack([b['goal_obs'] for b in batch], dim=0)
        # add discount
        B = next(iter(out.values())).shape[0]
        out['discount'] = torch.full((B, 1), fill_value=cfg.discount, dtype=torch.float32)
        return out
    return _collate


class Workspace:
    def __init__(self, cfg: Config) -> None:
        self.work_dir = Path.cwd()
        print(f"Workspace: {self.work_dir}")
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available() and cfg.device != "cpu":
            cfg.device = "cpu"
            cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        # build dataset and dataloader
        assert cfg.load_replay_buffer is not None, "load_replay_buffer (dataset root) must be provided"
        dataset = HilbertRepresentationDataset(
            data_dir=str(cfg.load_replay_buffer),
            types=cfg.hilbert_types,
            max_episodes_per_type=cfg.hilbert_max_episodes_per_type,
            goal_future=cfg.future,
            p_randomgoal=cfg.p_randomgoal,
            obs_horizon=cfg.hilbert_obs_horizon,
            full_loading=True,
            use_history_action=cfg.use_history_action,
        )
        self.dataset = dataset
        # infer dims
        obs_dim = int(dataset.obs_dim)
        # prefer dataset.action_dim when present, else fallback
        action_dim = int(dataset.action_dim) if dataset.action_dim is not None else 19

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        if cfg.use_wandb:
            import wandb
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=f"sd{cfg.seed:03d}_sf_dl",
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
                       dir=wandb_output_dir)

        # build agent
        agent_cfg = cfg.agent
        agent_cfg.obs_shape = (obs_dim,)
        self.agent = make_agent(obs_dim, action_dim, agent_cfg)

        # dataloader
        num_workers = int(getattr(agent_cfg, 'dataloader_num_workers', 4))
        pin_memory = bool(getattr(agent_cfg, 'dataloader_pin_memory', True))
        drop_last = bool(getattr(agent_cfg, 'dataloader_drop_last', True))
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=legged_collate(cfg),
        )
        self.data_iter = iter(self.loader)

        self.timer = utils.Timer()
        self.global_step = 0
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        self._checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def train(self) -> None:
        train_until_step = utils.Until(self.cfg.num_grad_steps)
        log_every_step = utils.Every(self.cfg.log_every_steps)
        while train_until_step(self.global_step):
            # update with dataloader iterator
            metrics = self.agent.update(self.data_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', self.cfg.log_every_steps / elapsed_time)
                    log('total_time', total_time)
                    log('step', self.global_step)
            self.global_step += 1

            # save rolling checkpoint without dataloader
            if self.cfg.num_grad_steps > 0 and self.global_step % max(1, self.cfg.log_every_steps * 10) == 0:
                self.save_checkpoint(self._checkpoint_filepath)

        self.save_checkpoint(self._checkpoint_filepath)

    _CHECKPOINTED_KEYS = ('agent', 'global_step')

    def save_checkpoint(self, fp: tp.Union[Path, str]) -> None:
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS}
        fp = Path(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, fp)


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    ws = Workspace(cfg)
    ws.train()


if __name__ == '__main__':
    main()

import platform
import os

# if 'mac' in platform.platform():
#     # macOS 下通常不需要特殊的渲染后端设置
#     pass
# else:
#     # 非 macOS：指定使用 EGL 作为 MuJoCo 的 GL 后端，以便无显示环境下渲染
#     os.environ['MUJOCO_GL'] = 'egl'
#     if 'SLURM_STEP_GPUS' in os.environ:
#         # 在 SLURM 作业环境下，将 EGL 使用的设备与分配的 GPU 对齐
#         os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

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

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=DeprecationWarning)  # 屏蔽冗余的弃用告警

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
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.hilbert_dataset import LeggedGymReplayLoader
from gym import spaces
from url_benchmark.replay_buffer import DataBuffer
from isaacgym import gymapi
from url_benchmark.legged_gym_env_utils import build_isaac_namespace, _to_rgb_frame

def _safe_set_viewer_cam(env, pos, look, track_index=0):
    env.set_camera(pos, look, track_index)

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
    example_replay_path: str = "/root/workspace/HugWBC/example_trajectories"
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
    headless: bool = False
    use_history_action: bool = True
    
    # eval
    num_eval_episodes: int = 10
    eval_every_steps: int = 10000
    num_final_eval_episodes: int = 50
    video_every_steps: int = 100000
    num_skip_frames: int = 2
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = ()
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_grad_steps: int = 1000000
    log_every_steps: int = 1000
    num_seed_frames: int = 0
    replay_buffer_episodes: int = 5000  # 从离线缓冲区加载的 episode 数上限
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    goal_eval: bool = False
    # dataset
    load_replay_buffer: tp.Optional[str] = None
    expl_agent: str = "rnd"
    replay_buffer_dir: str = omgcf.SI("../../../../datasets")
    # legged-gym dataset (Hilbert zarr) options
    hilbert_types: tp.Optional[tp.List[str]] = None
    hilbert_max_episodes_per_type: tp.Optional[int] = None
    hilbert_obs_horizon: int = 5
    # eval control
    eval_only: bool = False


ConfigStore.instance().store(name="workspace_config", node=Config)


def make_agent(
        obs_type: str, image_wh, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent, agents.DDPGAgent]:
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
        self.work_dir = Path.cwd()
        # if cfg.load_model is not None:
        #     self.work_dir = os.path.dirname(os.path.dirname(cfg.load_model))
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        task = cfg.task
        self.domain = task.split('_', maxsplit=1)[0]

        # self.train_env = self._make_env()  # 环境仅用于读取规格与评估
        self.eval_env = self._make_eval_env()
        self._init_env_cam(self.eval_env)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        if cfg.use_wandb:
            exp_name = ''
            exp_name += f'sd{cfg.seed:03d}_'
            if 'SLURM_JOB_ID' in os.environ:
                exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
            if 'SLURM_PROCID' in os.environ:
                exp_name += f'{os.environ["SLURM_PROCID"]}.'
            exp_name += '_'.join([cfg.agent.name, self.domain, str(self.cfg.discount), str(self.cfg.future), str(self.cfg.p_randomgoal), str(self.cfg.agent.hilp_expectile), str(self.cfg.agent.hilp_discount), str(self.cfg.agent.q_loss), str(self.cfg.agent.command_injection), str(self.cfg.agent.mix_ratio), str(self.cfg.use_history_action)
            ])

            wandb_output_dir = tempfile.mkdtemp()
            cfg_dict = omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
            cfg_dict['work_dir'] = self.work_dir
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                       config=cfg_dict,
                       dir=wandb_output_dir)
        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []

        print("loading Replay from %s", self.cfg.load_replay_buffer)
        hard_coded_act_spec = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        action_dim = hard_coded_act_spec.shape[0]
        self.replay_loader = LeggedGymReplayLoader(
            data_dir=str(cfg.load_replay_buffer),
            action_dim=int(action_dim),
            discount=float(cfg.discount),
            future=float(cfg.future),
            p_currgoal=float(cfg.p_currgoal),
            p_randomgoal=float(cfg.p_randomgoal),
            obs_horizon=int(cfg.hilbert_obs_horizon),
            types=cfg.hilbert_types,
            max_episodes_per_type=cfg.hilbert_max_episodes_per_type,
            full_loading=True,
            use_history_action=cfg.use_history_action,
        )

        sample = self.replay_loader.sample(1)
        print("Sample obs dim: ", sample.next_obs.shape[-1])
        flatten_obs_dim = int(sample.next_obs.shape[-1])
        self.flatten_obs_dim = flatten_obs_dim
        self.obs_dim = int(flatten_obs_dim // self.cfg.hilbert_obs_horizon)
        print("Obs dim per time step: ", self.obs_dim)
        agent_cfg = self.cfg.agent
        agent_cfg.obs_shape = (flatten_obs_dim,)
        self.agent = make_agent(cfg.obs_type,
                                cfg.image_wh,
                                specs.Array(shape=(self.flatten_obs_dim,), dtype=np.float32, name='obs'),
                                hard_coded_act_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                agent_cfg)
        self.example_data_buffers = {}
        for dirn in os.listdir(self.cfg.example_replay_path):
            full_path = os.path.join(self.cfg.example_replay_path, dirn)
            commandn = dirn.split('.')[0]
            example_data_buffer = DataBuffer.copy_from_path(full_path)
            self.example_data_buffers[commandn] = example_data_buffer
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None and cfg.load_model.endswith('.pt'):
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])
    
    def _make_eval_env(self):
        cfg = self.cfg
        # Only support legged-gym env here
        task_name = cfg.task
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=task_name)
        self.env_cfg.env.num_envs = 1
        self.env_cfg.env.episode_length_s = 1000

        # prevent in-episode command resampling; we will control commands manually
        self.env_cfg.commands.resampling_time = 1000

        # ---- Build an argparse.Namespace for Isaac Gym / legged-gym helpers ----
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

    # def _make_custom_reward(self) -> tp.Optional[BaseReward]:
    #     if self.cfg.custom_reward is None:
    #         return None
    #     return DmcReward(self.cfg.custom_reward)

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
        """主训练循环：反复从 ReplayBuffer 采样并调用 agent.update 进行训练，周期性评估与保存。"""
        if self.cfg.eval_only:
            if self.cfg.load_model is not None and not self.cfg.load_model.endswith('.pt'):
                ckpts = os.listdir(self.cfg.load_model)
                ckpts = [ckpt for ckpt in ckpts if ckpt.endswith('.pt')]
                ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                for ckpt in ckpts:
                    self.load_checkpoint(os.path.join(self.cfg.load_model, ckpt), exclude=["replay_loader"])
                    self.eval(self.global_step)
            else:
                self.eval(self.global_step)
            return

        train_until_step = utils.Until(self.cfg.num_grad_steps)
        eval_every_step = utils.Every(self.cfg.eval_every_steps)
        log_every_step = utils.Every(self.cfg.log_every_steps)

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_step)
                self.eval()

            metrics = self.agent.update(self.replay_loader, self.global_step)
            # wandb.log(metrics, step=self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', self.cfg.log_every_steps / elapsed_time)
                    log('total_time', total_time)
                    log('step', self.global_step)
            self.global_step += 1
            # try to save snapshot
            if self.global_frame in self.cfg.snapshot_at:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'),
                                     exclude=["replay_loader"])
            # save checkpoint to reload
            if self.cfg.checkpoint_every != 0 and self.global_frame % self.cfg.checkpoint_every == 0:
                _checkpoint_filepath = self.work_dir / "models" / f"{self.global_step}.pt"
                self.save_checkpoint(_checkpoint_filepath, exclude=["replay_loader"])
        if self.cfg.checkpoint_every != 0:
            self.save_checkpoint(self._checkpoint_filepath, exclude=["replay_loader"])  # make sure we save the final checkpoint
            
    def _proprocess_obs(self, obs):
        pure_obs = obs[..., :self.obs_dim].reshape(1, -1)                
        if self.cfg.use_history_action:
            obs_command = obs[:, -1, self.obs_dim:self.obs_dim + 11].reshape(1, -1)
            assert self.obs_dim + 13 == obs.shape[-1], f"obs_dim + 32 != obs.shape[-1], raw_obs_dim: {obs.shape[-1]} obs_dim: {self.obs_dim} sum: {self.obs_dim + 13} != {obs.shape[-1]}"
        else:
            obs_command = obs[:, -1, self.obs_dim + 19:self.obs_dim + 30].reshape(1, -1)
            assert self.obs_dim + 32 == obs.shape[-1], f"obs_dim + 13 != obs.shape[-1], raw_obs_dim: {obs.shape[-1]} obs_dim: {self.obs_dim} sum: {self.obs_dim + 32} != {obs.shape[-1]}"
        return pure_obs, obs_command

    def eval(self):
        commands_horizons = [10, 20, 40, 80, 160]
        eval_time = 10
        video_save_parent = self.work_dir / "models" / f"{self.global_step}"
        os.makedirs(video_save_parent, exist_ok=True)
        eval_steps = int(eval_time / self.eval_env.dt)
        eval_results = {'env_rewards': [], 'z_rewards': [], 'env_rewards_raw_cmd': [], 'z_rewards_raw_cmd': [], 'ep_len': [], 'ep_len_raw_cmd': []}
        if self.cfg.agent.command_injection:
            for key, data_buffer in self.example_data_buffers.items():
                command_name = key  
                env_commands = data_buffer.meta['episode_command_A'][:]
                random_traj_id = np.random.randint(0, len(env_commands))
                command_vec = torch.tensor(env_commands[random_traj_id], device=self.eval_env.device)
                VIDEO_PATH = video_save_parent / f"{command_name}_{random_traj_id}_raw_cmd.mp4"
                rewards_json_path = video_save_parent / f"{command_name}_{random_traj_id}_raw_cmd.json"
                _, _ = self.eval_env.reset()

                self.eval_env.commands[:, :10] = command_vec
                obs, critic_obs, _, _, _ = self.eval_env.step(torch.zeros(
                    self.eval_env.num_envs, self.eval_env.num_actions, dtype=torch.float, device=self.eval_env.device))
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

                timestep = 0
                pure_obs, obs_command = self._proprocess_obs(obs)
                last_index = -1

                # imageio 写 mp4，依赖 ffmpeg（imageio-ffmpeg）
                writer = imageio.get_writer(VIDEO_PATH, fps=self.FPS)
                dones = np.zeros(self.eval_env.num_envs, dtype=np.bool_)
                rewards_recorder = {}
                rewards_recorder['env_rewards'] = 0
                rewards_recorder['z_rewards'] = 0
                rewards_recorder['env_rew_list'] = []
                rewards_recorder['z_rew_list'] = []
                while not dones.any() and timestep < eval_steps:
                    with torch.inference_mode():
                        assert self.agent.command_injection, f"command_injection must be True"
                        z_vector = self.agent.sample_z(1, obs_command)
                        actions, _ = self.agent.actor.act_inference(pure_obs, z_vector)
                    last_obs = pure_obs
                    obs, critic_obs, reward, dones, _ = self.eval_env.step(actions)
                    
                    pure_obs, obs_command = self._proprocess_obs(obs)
                    self.eval_env.commands[:, :10] = command_vec
                    with torch.inference_mode():
                        z_reward = self.agent.get_z_rewards(last_obs, pure_obs, z_vector.cpu().numpy())
                    rewards_recorder['env_rew_list'].append(float(reward.item()))
                    rewards_recorder['z_rew_list'].append(float(z_reward))
                    rewards_recorder['env_rewards'] += reward.item()
                    rewards_recorder['z_rewards'] += z_reward
                    # print(f"z_reward: {z_reward}, env_reward: {reward.item()}")
                    if dones.any():
                        print(f"command: {command_name}, episode env reward: {reward.item()}, episode z reward: {rewards_recorder['z_rewards']}, ep_len: {len(rewards_recorder['env_rew_list'])}")
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

                    # ===== 干扰和中断设置 =====
                    self.eval_env.use_disturb = True
                    self.eval_env.disturb_masks[:] = True
                    self.eval_env.disturb_isnoise[:] = True
                    self.eval_env.disturb_rad_curriculum[:] = 1.0
                    self.eval_env.interrupt_mask[:] = self.eval_env.disturb_masks[:]
                    self.eval_env.standing_envs_mask[:] = True

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
                    timestep += 1
                writer.close()
                with open(rewards_json_path, 'w') as f:
                    json.dump(rewards_recorder, f, indent=4)
                print("Saved video to %s", VIDEO_PATH)
                mean_env_rewards = np.mean(rewards_recorder['env_rew_list'])
                mean_z_rewards = np.mean(rewards_recorder['z_rew_list'])
                ep_len = len(rewards_recorder['env_rew_list'])
                metrics = {f"{command_name}_env_rew_raw_cmd": mean_env_rewards, f"{command_name}_z_rew_raw_cmd": mean_z_rewards, f"{command_name}_ep_len": ep_len}
                eval_results['env_rewards_raw_cmd'].append(mean_env_rewards)
                eval_results['z_rewards_raw_cmd'].append(mean_z_rewards)
                eval_results['ep_len_raw_cmd'].append(ep_len)
                self.logger.log_metrics(metrics, self.global_step, ty='eval_detail')
        for command_horizon in commands_horizons:
            horizon_results = {'env_rewards': [], 'z_rewards': [], 'ep_len': []}
            for key, data_buffer in self.example_data_buffers.items():
                command_name = key
                env_commands = data_buffer.meta['episode_command_A'][:]
                random_traj_id = np.random.randint(0, len(env_commands))
                command_vec = torch.tensor(env_commands[random_traj_id], device=self.eval_env.device)
                VIDEO_PATH = video_save_parent / f"{command_name}_{random_traj_id}_{command_horizon}.mp4"
                rewards_json_path = video_save_parent / f"{command_name}_{random_traj_id}_{command_horizon}.json"
                data_start_idx = 0 if random_traj_id == 0 else data_buffer.episode_ends[random_traj_id - 1]
                data_end_idx = data_buffer.episode_ends[random_traj_id]
                len_trajectory = data_end_idx - data_start_idx
                _, _ = self.eval_env.reset()
                obs, critic_obs, _, _, _ = self.eval_env.step(torch.zeros(
                    self.eval_env.num_envs, self.eval_env.num_actions, dtype=torch.float, device=self.eval_env.device))
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
                self.eval_env.commands[:, :10] = command_vec


                z_resample_interval = command_horizon
                resample_steps_list = [i for i in range(0, len_trajectory + z_resample_interval - 1, z_resample_interval)]
                goals_list = [data_buffer.data['proprio'][i, :, :self.obs_dim].reshape(1, -1) for i in resample_steps_list]
                goals_list.append(data_buffer.data['proprio'][-1, :, :self.obs_dim].reshape(1, -1))

                timestep = 0
                pure_obs, obs_command = self._proprocess_obs(obs)
                last_index = -1

                # imageio 写 mp4，依赖 ffmpeg（imageio-ffmpeg）
                writer = imageio.get_writer(VIDEO_PATH, fps=self.FPS)
                dones = np.zeros(self.eval_env.num_envs, dtype=np.bool_)
                rewards_recorder = {}
                rewards_recorder['env_rewards'] = 0
                rewards_recorder['z_rewards'] = 0
                rewards_recorder['env_rew_list'] = []
                rewards_recorder['z_rew_list'] = []
                while not dones.any() and timestep < eval_steps:
                    with torch.inference_mode():
                        if timestep // z_resample_interval != last_index and last_index < len(goals_list) - 1:
                            print(f"Switch goal state: {last_index} -> {timestep // z_resample_interval} / {eval_steps}")
                            last_index = timestep // z_resample_interval
                            last_index = min(last_index, len(goals_list) - 1)
                        meta = self.agent.get_goal_meta(goal_array=goals_list[last_index].squeeze(0), obs_array=pure_obs.squeeze(0))
                        z_vector = torch.tensor(meta['z'], device=self.eval_env.device).reshape(1, -1)
                        actions, _ = self.agent.actor.act_inference(pure_obs, z_vector)
                    last_obs = pure_obs
                    obs, critic_obs, reward, dones, _ = self.eval_env.step(actions)
                    
                    pure_obs, obs_command = self._proprocess_obs(obs)
                    self.eval_env.commands[:, :10] = command_vec
                    with torch.inference_mode():
                        z_reward = self.agent.get_z_rewards(last_obs, pure_obs, z_vector.cpu().numpy())
                    rewards_recorder['env_rew_list'].append(float(reward.item()))
                    rewards_recorder['z_rew_list'].append(float(z_reward))
                    rewards_recorder['env_rewards'] += reward.item()
                    rewards_recorder['z_rewards'] += z_reward
                    # print(f"z_reward: {z_reward}, env_reward: {reward.item()}")
                    if dones.any():
                        print(f"command: {command_name}, episode env reward: {reward.item()}, episode z reward: {rewards_recorder['z_rewards']}, ep_len: {len(rewards_recorder['env_rew_list'])}")
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

                    # ===== 干扰和中断设置 =====
                    self.eval_env.use_disturb = True
                    self.eval_env.disturb_masks[:] = True
                    self.eval_env.disturb_isnoise[:] = True
                    self.eval_env.disturb_rad_curriculum[:] = 1.0
                    self.eval_env.interrupt_mask[:] = self.eval_env.disturb_masks[:]
                    self.eval_env.standing_envs_mask[:] = True

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
                    timestep += 1
                writer.close()
                with open(rewards_json_path, 'w') as f:
                    json.dump(rewards_recorder, f, indent=4)
                print("Saved video to %s", VIDEO_PATH)
                mean_env_rewards = np.mean(rewards_recorder['env_rew_list'])
                mean_z_rewards = np.mean(rewards_recorder['z_rew_list'])
                ep_len = len(rewards_recorder['env_rew_list'])
                metrics = {f"{command_name}_{command_horizon}_env_rew": mean_env_rewards, f"{command_name}_{command_horizon}_z_rew": mean_z_rewards, f"{command_name}_{command_horizon}_ep_len": ep_len}
                eval_results['env_rewards'].append(mean_env_rewards)
                eval_results['z_rewards'].append(mean_z_rewards)
                eval_results['ep_len'].append(ep_len)
                self.logger.log_metrics(metrics, self.global_step, ty='eval_detail')
                horizon_results['env_rewards'].append(mean_env_rewards)
                horizon_results['z_rewards'].append(mean_z_rewards)
                horizon_results['ep_len'].append(ep_len)
            metrics = {f"horizon_{command_horizon}_env_rew": np.mean(horizon_results['env_rewards']), f"horizon_{command_horizon}_z_rew": np.mean(horizon_results['z_rewards']), f"horizon_{command_horizon}_ep_len": np.mean(horizon_results['ep_len'])}
            self.logger.log_metrics(metrics, self.global_step, ty='eval_horizon')
        metrics = {key: np.mean(eval_results[key]) for key in eval_results if len(eval_results[key]) > 0}
        self.logger.log_metrics(metrics, self.global_step, ty='eval_mean')
        
        # 确保所有eval层级的metrics都被dump到wandb
        self.logger.dump_all_eval(self.global_step)

          

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        """保存关键状态用于断点重训。

        保存内容包含：agent、global_step、global_episode、replay_loader（可被 only/exclude 调整）。
        """
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        if "replay_loader" not in exclude:
            assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = (), num_episodes=None, use_pixels=False) -> None:
        """从磁盘加载 checkpoint。

        - only：仅恢复指定键（例如只加载 replay_loader）。
        - exclude：排除指定键（例如不恢复 replay_loader）。
        - num_episodes：可在加载时裁剪 ReplayBuffer 的 episode 数。
        - use_pixels：当使用像素观测时，将存储中的 'pixel' 字段重命名为 'observation'。
        """
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)

        if num_episodes is not None:
            payload._episodes_length = payload._episodes_length[:num_episodes]
            payload._max_episodes = min(payload._max_episodes, num_episodes)
            for key, value in payload._storage.items():
                payload._storage[key] = value[:num_episodes]
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')

        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                self.replay_loader = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
