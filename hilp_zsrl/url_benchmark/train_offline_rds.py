import platform
import os

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
import time
import logging
import torch
import warnings
import plotly.graph_objects as go
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)
import dataclasses
import tempfile
import typing as tp
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
import torch
import wandb
import omegaconf as omgcf
from url_benchmark.dataset_utils.utils import InfiniteDataLoaderWrapper
from url_benchmark.utils import utils
from url_benchmark.utils.video import VideoRecorder
from tqdm import trange
from url_benchmark.dmc_utils.gym_vector_env import make_gym_async_vectorized
from url_benchmark.train_online import Config, Workspace, make_agent

@dataclasses.dataclass
class RDSConfig(Config):
    agent: tp.Any
    # misc
    run_group: str = "Debug"
    seed: int = 1
    device: str = "cuda"
    save_video: bool = True
    use_tb: bool = False
    use_wandb: bool = True
    # experiment
    experiment: str = "offline"
    # task settings
    task: str = "walker_run"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    image_wh: int = 64
    action_repeat: int = 1
    discount: float = 0.98
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    p_currgoal: float = 0  # current goal ratio
    p_randomgoal: float = 0  # random goal ratio
    num_episode_steps: int = 1000
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
    local_replay_buffer_path: tp.Optional[str] = None
    # training
    sac_optim_steps: int = 50
    phi_net_pretrain_steps: int = 80000
    num_workers: int = 2
    rollout_every_steps: int = 20000
    save_every_steps: int = 20000
    phi_total_episodes: int = 100000
    phi_rollout_num: int = 100
    num_train_envs: int = 100
    num_grad_steps: int = 100000000
    log_every_steps: int = 1000
    num_seed_frames: int = 0
    replay_buffer_init_size: int = 1000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    goal_eval: bool = False
    # dataset
    load_replay_buffer: tp.Optional[str] = None
    expl_agent: str = "rnd"
    replay_buffer_dir: str = omgcf.SI("../../../../datasets")
    resume_from: tp.Optional[str] = None
    eval_only: bool = False

ConfigStore.instance().store(name="workspace_config", node=RDSConfig)

class RDSWorkspace(Workspace):
    def __init__(self, cfg: RDSConfig) -> None:
        if cfg.resume_from is not None:
            self.work_dir = Path(cfg.resume_from)
        else:
            self.work_dir = Path.cwd()
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

        self.train_env = make_gym_async_vectorized(
            name=self.cfg.task,
            num_envs=self.cfg.num_train_envs,
            obs_type=self.cfg.obs_type,
            frame_stack=self.cfg.frame_stack,
            action_repeat=self.cfg.action_repeat,
            seed=self.cfg.seed,
            image_wh=self.cfg.image_wh,
        )

        self.eval_env = self._make_env()
        # create agent
        # self.train_env.reset()
        self.agent = make_agent(cfg.obs_type,
                                cfg.image_wh,
                                self.train_env.observation_space,
                                self.train_env.action_space,
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        if cfg.use_wandb:
            exp_name = ''
            exp_name += f'sd{cfg.seed:03d}_'
            if 'SLURM_JOB_ID' in os.environ:
                exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
            if 'SLURM_PROCID' in os.environ:
                exp_name += f'{os.environ["SLURM_PROCID"]}.'
            exp_name += '_'.join([
                cfg.run_group, cfg.agent.name, self.domain,
            ])
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                       dir=wandb_output_dir)

        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, task=self.cfg.task,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.phi_step = 0
        self.policy_step = 0
        self.global_step = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["phi_dataset"])

    
    def train(self):
        if self.cfg.eval_only:
            self.eval_sum(self.phi_dataset)
            return
        self.global_progress_bar = trange(self.cfg.num_grad_steps, position=0, initial=self.global_step, leave=True, desc="Training Global")
        if not hasattr(self, 'phi_dataloader'):
            self.prepare_phi_dataset(is_init=True)
        while True:
            self.eval_sum(self.phi_dataset)
            self.train_policy()
            if self.global_step >= self.cfg.num_grad_steps:
                print("Training completed.")
                break


    def prepare_phi_dataset(self, is_init=False):
        self.init_phi_dataset()
        start_time = time.time()
        if self.cfg.local_replay_buffer_path is not None:
            rp_cls = torch.load(self.cfg.local_replay_buffer_path)
            episode_data = {}
            episode_data['observation'] = rp_cls._storage['observation'][:, :-1]
            episode_data['action'] = rp_cls._storage['action'][:, 1:]
            episode_data['reward'] = rp_cls._storage['reward'][:, 1:]
            # episode_data['discount'] = rp_cls._storage['discount'][1:]
            episode_data['physics'] = rp_cls._storage['physics'][:, 1:]
            self.phi_dataset.add_episodes(episode_data, num_episodes=rp_cls._storage['observation'].shape[0])
        else:
            total_num = self.cfg.phi_rollout_num
            pbar = trange(total_num, desc="Collecting data", leave=True)
            for _ in range(total_num):
                episodes = self._collect_episodes(random_sample=True, is_init=is_init)
                pbar.update(1)
                self.phi_dataset.add_episodes(episodes, num_episodes=self.cfg.num_train_envs)
                pbar.set_description(f"Steps: {len(self.phi_dataset)}")
        dataloader_cfg = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": self.cfg.num_workers,
        }
        self.phi_dataloader = InfiniteDataLoaderWrapper(self.phi_dataset, dataloader_cfg)
        end_time = time.time()
        if self.cfg.use_wandb:
            wandb.log({f"train/phi_dataset_time": end_time - start_time, 'phi_dataset_ep_num': self.phi_dataset.meta_info['current_size']}, step=self.global_step)
        else:
            print(f"train/phi_dataset_time: {end_time - start_time}")

    def train_policy(self):
        for batch in self.phi_dataloader:
            metrics = self.agent.update_batch(batch, update_all=True)
            self.global_progress_bar.update(1)
            self.global_progress_bar.set_description(f"Training")
            metrics['phi_dataset_steps'] = len(self.phi_dataset)
            self.update_train_metrics(metrics)
            self.global_step += 1
            if self.global_step % self.cfg.save_every_steps == 0:
                _checkpoint_filepath = os.path.join(self.work_dir, "models", f"{self.global_step}.pt")
                self.save_checkpoint(_checkpoint_filepath)
            if self.global_step % self.cfg.rollout_every_steps == 0:
                return

@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = RDSWorkspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()