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
from collections import defaultdict

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

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        if cfg.use_wandb:
            exp_name = 'phi_only'
            exp_name += f'sd{cfg.seed:03d}_'
            if 'SLURM_JOB_ID' in os.environ:
                exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
            if 'SLURM_PROCID' in os.environ:
                exp_name += f'{os.environ["SLURM_PROCID"]}.'
            exp_name += '_'.join([cfg.agent.name, self.domain, str(self.cfg.discount), str(self.cfg.future), str(self.cfg.p_randomgoal), str(self.cfg.agent.hilp_expectile), str(self.cfg.agent.hilp_discount),
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
            full_loading=False,
            split_val_set=True,
        )

        sample = self.replay_loader.sample(1)
        flatten_obs_dim = int(sample.next_obs.shape[-1])
        self.flatten_obs_dim = flatten_obs_dim
        self.obs_dim = int(flatten_obs_dim // self.cfg.hilbert_obs_horizon)
        agent_cfg = self.cfg.agent
        agent_cfg.obs_shape = (flatten_obs_dim,)
        self.agent = make_agent(cfg.obs_type,
                                cfg.image_wh,
                                specs.Array(shape=(flatten_obs_dim,), dtype=np.float32, name='obs'),
                                hard_coded_act_spec,
                                cfg.num_seed_frames // cfg.action_repeat,
                                agent_cfg)
        self.example_data_buffers = {}
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None and cfg.load_model.endswith('.pt'):
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

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

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.global_step):
                # self.logger.log('eval_total_time', self.timer.total_time(), self.global_step)
                val_metrics = defaultdict(list)
                for i in range(100):
                    metrics = self.agent.update(self.replay_loader, self.global_step, is_val=True)
                for k, v in metrics.items():
                    val_metrics[k].append(v)
                for k, v in val_metrics.items():
                    val_metrics[k] = np.mean(v)
                wandb.log(val_metrics, step=self.global_step)

            metrics = self.agent.update(self.replay_loader, self.global_step)
            wandb.log(metrics, step=self.global_step)
            # self.logger.log_metrics(metrics, self.global_step, ty='train')
            # if log_every_step(self.global_step):
            #     elapsed_time, total_time = self.timer.reset()
            #     with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
            #         log('fps', self.cfg.log_every_steps / elapsed_time)
            #         log('total_time', total_time)
            #         log('step', self.global_step)
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
