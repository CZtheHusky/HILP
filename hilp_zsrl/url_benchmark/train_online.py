import platform
import os
from dm_control.suite.wrappers import action_scale
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
import math
import shutil
import time
import logging
import torch
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)
from collections import defaultdict
import json
import dataclasses
from collections import deque
import tempfile
import typing as tp
from pathlib import Path
import torch.nn.functional as F
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf
from url_benchmark.dataset_utils.phi_pretrain_dataset import PhiWalkerDataset
from url_benchmark.dataset_utils.utils import InfiniteDataLoaderWrapper
from url_benchmark.dmc_utils import dmc
from dm_env import specs
from url_benchmark.utils import utils
from url_benchmark import agent as agents
from url_benchmark.utils.logger import Logger
from url_benchmark.dataset_utils.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.utils.video import VideoRecorder
from url_benchmark.utils.my_utils import record_video
from tqdm import trange
import traceback
from url_benchmark.utils.humanoid_utils import _safe_set_viewer_cam, get_cosine_sim, collect_nonzero_losses, calc_phi_loss_upper, calc_z_vector_matrixes, plot_matrix_heatmaps, compute_global_color_limits, plot_tripanel_heatmaps_with_line, plot_per_step_z
from url_benchmark.dmc_utils.gym_vector_env import make_gym_async_vectorized

def generate_cosine_variant(phi_goal: np.ndarray, cos_target: float, device: str = 'cuda') -> tp.Tuple[torch.Tensor, torch.Tensor]:
    base_tensor = torch.as_tensor(phi_goal, dtype=torch.float32, device=device)
    norm = torch.norm(base_tensor)
    if norm < 1e-8:
        # Degenerate input: fall back to a fixed direction and return a valid (phi_goal, z_actor)
        fallback = torch.zeros_like(base_tensor)
        if fallback.numel() > 0:
            fallback.view(-1)[0] = 1.0
        z_actor = math.sqrt(fallback.shape[-1]) * F.normalize(fallback, dim=-1)
        return fallback, z_actor
    unit_base = base_tensor / norm
    if abs(abs(cos_target) - 1.0) < 1e-6:
        base_tensor = base_tensor * (1.0 if cos_target >= 0 else -1.0)
        z_actor = math.sqrt(base_tensor.shape[-1]) * F.normalize(base_tensor, dim=-1)
        return base_tensor, z_actor
    ortho = torch.randn_like(base_tensor)
    ortho -= torch.dot(ortho, unit_base) * unit_base
    ortho_norm = torch.norm(ortho)
    if ortho_norm < 1e-8:
        ortho = torch.zeros_like(base_tensor)
        ortho[0] = 1.0
        ortho -= torch.dot(ortho, unit_base) * unit_base
        ortho_norm = torch.norm(ortho)
    ortho_unit = ortho / ortho_norm
    sin_component = math.sqrt(max(0.0, 1.0 - cos_target ** 2))
    rotated = cos_target * unit_base + sin_component * ortho_unit
    phi_goal = rotated * norm
    z_actor = math.sqrt(phi_goal.shape[-1]) * F.normalize(phi_goal, dim=-1)
    return phi_goal, z_actor

def rescale_to_unit_range(data_list):
    """Rescale each episode's data to [0, 1] range for comparison"""
    rescaled_data = []
    for episode_data in data_list:
        if len(episode_data) == 0:
            rescaled_data.append([])
            continue
        episode_array = np.array(episode_data)
        min_val = episode_array.min()
        max_val = episode_array.max()
        if max_val - min_val > 1e-8:  # avoid division by zero
            rescaled = (episode_array - min_val) / (max_val - min_val)
        else:
            rescaled = np.zeros_like(episode_array)
        rescaled_data.append(rescaled.tolist())
    return rescaled_data

def log_plotly_lines(prefix, ep, xs, series, global_frame):
    fig = go.Figure()
    for name, ys in series.items():
        L = min(len(xs), len(ys))
        fig.add_trace(go.Scatter(x=xs[:L], y=ys[:L], mode="lines", name=name))
    fig.update_layout(title=f"{prefix} Episode {ep}: Rescaled Latent vs Reward",
                      xaxis_title="step", yaxis_title="value")
    wandb.log({f"{prefix}/combined_rew_scaled_ep{ep}": fig}, step=global_frame)

def plot_wandb_lines(data_lists, prefix, global_frame, key_name="raw_dist_ep", x_name="step", y_name="raw_dist"):
    xs = [list(range(len(data))) for data in data_lists]
    ys = data_lists
    plot_keys = [f'{prefix}/{key_name}{ep}' for ep in range(len(data_lists))]
    wandb_tables = [
        wandb.Table(data=list(zip(xs[ep], ys[ep])), columns=[x_name, y_name]) for ep in range(len(data_lists))
    ]
    charts_dict = {f"{prefix}/{key_name}{ep}": wandb.plot.line(
        wandb_tables[ep], x_name, y_name, title=plot_keys[ep]) for ep in range(len(data_lists)
        )}
    wandb.log(charts_dict, step=global_frame)

@dataclasses.dataclass
class Config:
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
    # training
    sac_optim_steps: int = 50
    phi_net_pretrain_steps: int = 50000
    num_workers: int = 2
    rollout_every_steps: int = 50000
    save_every_steps: int = 50000
    phi_total_episodes: int = 100000
    phi_rollout_num: int = 100
    num_train_envs: int = 100
    num_grad_steps: int = 100000000
    log_every_steps: int = 1000
    num_seed_frames: int = 0
    replay_buffer_episodes: int = 100000
    replay_buffer_init_size: int = 10000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    goal_eval: bool = False
    # dataset
    replay_buffer_dir: str = omgcf.SI("../../../../datasets")
    resume_from: tp.Optional[str] = None
    eval_only: bool = False
    extensive_eval: bool = False


ConfigStore.instance().store(name="workspace_config", node=Config)


class BaseReward:
    def __init__(self, seed: tp.Optional[int] = None) -> None:
        self._env: dmc.EnvWrapper  # to be instantiated in subclasses
        self._rng = np.random.RandomState(seed)

    def get_goal(self, goal_space: str) -> np.ndarray:
        raise NotImplementedError

    def from_physics(self, physics: np.ndarray) -> float:
        "careful this is not threadsafe"
        with self._env.physics.reset_context():
            self._env.physics.set_state(physics)
        return self.from_env(self._env)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        raise NotImplementedError


class DmcReward(BaseReward):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        env_name, task_name = name.split("_", maxsplit=1)
        from dm_control import suite  # import
        from url_benchmark import custom_dmc_tasks as cdmc
        if 'jaco' not in env_name:
            make = suite.load if (env_name, task_name) in suite.ALL_TASKS else cdmc.make
            self._env = make(env_name, task_name)
        else:
            self._env = cdmc.make_jaco(task_name, obs_type='states', seed=0)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        return float(self._env.task.get_reward(env.physics))


def make_agent(
        obs_type: str, image_wh, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.SFV2OnlineAgent, agents.SFAgent]:
    cfg.obs_type = obs_type
    cfg.image_wh = image_wh
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def _init_eval_meta(workspace, replay_loader, custom_reward: BaseReward = None, feature_type: str = None):
    num_steps = workspace.agent.cfg.num_inference_steps
    obs_list, reward_list, next_obs_list = [], [], []
    batch_size = 0
    while batch_size < num_steps:
        batch = replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
        batch = batch.to(workspace.cfg.device)
        obs_list.append(batch.obs)
        next_obs_list.append(batch.next_obs)
        reward_list.append(batch.reward)
        batch_size += batch.next_obs.size(0)
    obs, reward, next_obs = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
    obs_t, reward_t, next_obs_t = obs[:num_steps], reward[:num_steps], next_obs[:num_steps]
    return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t)


class Workspace:
    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'phi_step', "policy_step")
    def __init__(self, cfg: Config) -> None:
        if cfg.resume_from is not None:
            if cfg.resume_from.endswith('.pt'):
                self.work_dir = Path(cfg.resume_from).parent.parent
            else:
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
            config_dict = omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            config_dict['work_dir'] = self.work_dir
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                       config=config_dict,
                       dir=wandb_output_dir)

        self.replay_loader = ReplayBuffer(p_currgoal=cfg.p_currgoal, p_randomgoal=cfg.p_randomgoal, max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future, dummy_steps=0, is_mismatched=False)


        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, task=self.cfg.task,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.phi_step = 0
        self.policy_step = 0
        self.global_step = 0
        self.eval_rewards_history: tp.List[float] = []
        if cfg.resume_from is not None:
            if cfg.resume_from.endswith('.pt'):
                self._checkpoint_filepath = cfg.resume_from
            else:
                ckpt_path = self.work_dir / "models"
                models = [filen for filen in os.listdir(ckpt_path) if filen.endswith('.pt')]
                if len(models) == 1 and "latest" in models[0]:
                    self._checkpoint_filepath = os.path.join(self.work_dir, "models", "latest.pt")
                else:
                    models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    self._checkpoint_filepath = os.path.join(self.work_dir, "models", models[-1])
            if os.path.exists(self._checkpoint_filepath):
                self.load_checkpoint(self._checkpoint_filepath)

        self.replay_loader._frame_stack = cfg.frame_stack if cfg.obs_type == 'pixels' else None

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, image_wh=cfg.image_wh)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self) -> tp.Optional[BaseReward]:
        if self.cfg.custom_reward is None:
            return None
        return DmcReward(self.cfg.custom_reward)

    def get_argmax_goal_potential(self, replay_loader, custom_reward, return_indices: bool = False):
        """Sample transitions from replay and return the next_obs with maximum instantaneous reward.
        If return_indices is True, also return (ep_idx, step_idx) of that transition in the buffer.
        """
        num_steps = self.agent.cfg.num_inference_steps
        all_rewards = []
        all_next_obs = []
        all_ep_idx = []
        all_step_idx = []
        collected = 0
        while collected < num_steps:
            batch = replay_loader.sample_transitions_with_indices(self.cfg.batch_size, custom_reward=custom_reward)
            next_obs = batch['next_obs']
            reward = batch['reward']
            ep_idx = batch['ep_idx']
            step_idx = batch['step_idx']
            all_next_obs.append(next_obs)
            all_rewards.append(reward)
            all_ep_idx.append(ep_idx)
            all_step_idx.append(step_idx)
            collected += next_obs.shape[0]
        next_obs_arr = np.concatenate(all_next_obs, axis=0)[:num_steps]
        reward_arr = np.concatenate(all_rewards, axis=0)[:num_steps]
        ep_idx_arr = np.concatenate(all_ep_idx, axis=0)[:num_steps]
        step_idx_arr = np.concatenate(all_step_idx, axis=0)[:num_steps]
        # rewards are (N,1) -> (N,)
        reward_flat = reward_arr.reshape(-1)
        best = int(np.argmax(reward_flat))
        goal = next_obs_arr[best]
        best_reward = reward_flat[best]
        if return_indices:
            return goal, int(ep_idx_arr[best]), int(step_idx_arr[best]), best_reward
        return goal, best_reward


    def eval_sum(self, replay_loader):
        if self.cfg.extensive_eval:
            self.potential_eval_extensive(replay_loader)
        else:
            try:
                self.potential_eval(replay_loader, 'random')
            except Exception as e:
                print("Potential eval failed: random")
                print(traceback.format_exc())
            try:
                self.potential_eval(replay_loader, 'fit')
            except Exception as e:
                print("Potential eval failed: fit")
                print(traceback.format_exc())
            try:
                self.potential_eval(replay_loader, 'argmax')
            except Exception as e:
                print("Potential eval failed: argmax")
                print(traceback.format_exc())
    
    def train(self):
        if self.cfg.eval_only:
            if self.cfg.resume_from.endswith('.pt'):
                self.prepare_phi_dataset(is_init=False)
                self.eval_sum(self.phi_dataset)
            else:
                ckpt_path = self.work_dir / "models"
                models = [filen for filen in os.listdir(ckpt_path)]
                models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                for modeln in models:
                    self.load_checkpoint(os.path.join(self.work_dir, "models", modeln))
                    if hasattr(self, 'phi_dataset'):
                        del self.phi_dataset
                    self.prepare_phi_dataset(is_init=False)
                    self.eval_sum(self.phi_dataset)
            return
        if self.cfg.resume_from is None:
            self.train_phi(is_init=True)
        elif not self.cfg.resume_from.endswith('phi_pretrained_tmp.pt'):
            self.train_phi(is_init=False)
        self.global_progress_bar = trange(self.cfg.num_grad_steps, position=0, initial=self.global_step, leave=True, desc="Training Global")
        while True:
            self.eval_sum(self.phi_dataset)
            self.train_policy()
            self.train_phi()
            if self.global_step >= self.cfg.num_grad_steps:
                print("Training completed.")
                break

    @torch.no_grad()
    def potential_eval(self, replay_loader, eval_type='argmax'):
        """Goal-reaching evaluation.
        Pick a high-reward state from replay as goal, compute its (ep, step), then:
        - compare policy steps to reach that state vs step index in replay
        - log dataset partial return (up to that step) and policy return until reach
        """
        step, episode = 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        physics_agg = dmc.PhysicsAggregator()
        custom_reward = self._make_custom_reward()
        videos = []
        best_reward = None
        if eval_type == "argmax":
            current_goal_path = self.work_dir / self.cfg.task / 'potential_test'
            print("Argmax Goal evaluating")
        elif eval_type == "random":
            current_goal_path = self.work_dir / self.cfg.task / 'random_potential_test'
            print("Random goal evaluating")
        elif eval_type == "fit":
            meta, diag = _init_eval_meta(self, replay_loader, custom_reward, feature_type=eval_type)
            phi_goal = torch.as_tensor(meta['z_raw']).to(self.cfg.device)
            goal = None
            current_goal_path = self.work_dir / self.cfg.task / 'fit_potential_test'
            print("Fit goal evaluating")
        prefix = eval_type
        shutil.rmtree(current_goal_path, ignore_errors=True)
        current_goal_path.mkdir(exist_ok=True, parents=True)
        raw_obs_distances = [[] for _ in range(self.cfg.num_eval_episodes)]
        latent_distances = [[] for _ in range(self.cfg.num_eval_episodes)]
        task_reward = [[] for _ in range(self.cfg.num_eval_episodes)]
        ds_raw_reward = []
        ds_latent_dist = []
        all_rewards = []
        arg_max_rewards = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            if eval_type == 'argmax':
                # choose goal from replay with indices
                goal, ep_idx, step_idx, best_reward = self.get_argmax_goal_potential(replay_loader, custom_reward, return_indices=True)
                ep_idx, ds_trajectory, ds_rewards = replay_loader.get_episode(ep_idx, custom_reward)
                arg_max_rewards.append(best_reward)
            elif eval_type == 'random':
                # randomly pick an episode and use its final state as goal
                ep_idx, ds_trajectory, ds_rewards = replay_loader.get_episode(custom_reward=custom_reward)
                goal = ds_trajectory[-1]
                step_idx = len(ds_trajectory) - 1
            if eval_type != 'fit':
                ds_trajectory = torch.as_tensor(ds_trajectory[:step_idx + 1], device=self.cfg.device)
                ds_trajectory_phi = self.agent.feature_learner.feature_net(self.agent.encoder(ds_trajectory)).detach()
                ds_delta_phi = ds_trajectory_phi - ds_trajectory_phi[-1]
                ds_delta_phi = torch.norm(ds_delta_phi, dim=-1).cpu().numpy()
                ds_latent_dist.append(ds_delta_phi.tolist())
                meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)
                phi_goal = self.agent.feature_learner.feature_net(self.agent.encoder(torch.as_tensor(goal, device=self.cfg.device).unsqueeze(0))).detach()
                # dataset partial return up to goal step (inclusive)
                ds_partial_return = float(np.sum(ds_rewards[:step_idx+1]))
                ds_raw_reward.append(ds_rewards.reshape(-1).tolist())
                if self.cfg.use_wandb:
                    wandb.log({
                        f'{prefix}/ds_partial_return': ds_partial_return,
                    }, step=self.global_frame)
            # rollout aiming at goal
            total_reward = 0.0
            video_enabled = True
            self.video_recorder.init(self.eval_env, enabled=video_enabled)
            current_phi = []
            t = 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, phi = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                    current_phi.append(phi)
                # latent-space distance to goal
                try:
                    latent_dist = torch.norm(phi.view(-1) - phi_goal.view(-1)).item()
                except Exception as e:
                    latent_dist = float('nan')
                    print("Latent distance computation failed")
                    print(traceback.format_exc())
                latent_distances[episode].append(latent_dist)

                time_step = self.eval_env.step(action)
                physics_agg.add(self.eval_env)
                if step % self.cfg.num_skip_frames == 0:
                    self.video_recorder.record(self.eval_env)
                if custom_reward is not None:
                    time_step.reward = custom_reward.from_env(self.eval_env)
                total_reward += time_step.reward
                task_reward[episode].append(time_step.reward)
                if goal is not None:
                    raw_obs_distances[episode].append(float(np.linalg.norm(time_step.observation - goal)))
                t += 1
                step += 1
            if video_enabled:
                videos.append(self.video_recorder.frames)
            self.video_recorder.save(f'{prefix}_{self.global_frame}_{episode}.mp4')
            try:
                hilbert_traj = torch.cat(current_phi, axis=0)
                last_z = hilbert_traj[-1]
                z_diff = torch.diff(hilbert_traj, dim=0)
                z_diff_normed = math.sqrt(self.cfg.agent.z_dim) * F.normalize(z_diff, dim=1)
                z_diff_normed = z_diff_normed.cpu().numpy()
                goal_z_cosine_sim_list, goal_distance_list, goal_absdist_list = calc_z_vector_matrixes(hilbert_traj.cpu().numpy(), goal_vector=last_z.cpu().numpy())
                plot_tripanel_heatmaps_with_line(
                    goal_z_cosine_sim_list,
                    goal_distance_list,
                    goal_absdist_list,
                    [f"goal={goal_z_cosine_sim_list[g_idx].shape[-1]}_{episode}_{np.round(total_reward, 2)}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
                    current_goal_path,
                    title_cos=f'{episode} Z Cosine Similarity',
                    title_dist=f'{episode} Latent Space Distance',
                )
                current_z = torch.as_tensor(meta['z']).to(self.cfg.device).unsqueeze(0)
                goal_z_cosine_sim_list, goal_distance_list, goal_absdist_list = calc_z_vector_matrixes(hilbert_traj.cpu().numpy(), goal_vector=current_z.cpu().numpy())
                plot_tripanel_heatmaps_with_line(
                    goal_z_cosine_sim_list,
                    goal_distance_list,
                    goal_absdist_list,
                    [f"goal=z_argmax_{episode}_{np.round(total_reward, 2)}.png" for g_idx in range(len(goal_z_cosine_sim_list))],
                    current_goal_path,
                    title_cos=f'z_argmax {episode} Z Cosine Similarity',
                    title_dist=f'z_argmax {episode} Latent Space Distance',
                )
            except Exception as e:
                logger.warning(f"Potential eval plotting failed: {e}")
            all_rewards.append(total_reward)
            episode += 1
        all_rewards = np.array(all_rewards)
        if self.cfg.use_wandb:
            wandb.log({
                f'{prefix}/episode_reward': np.mean(all_rewards),
                f'{prefix}/episode_reward_std': np.std(all_rewards),
                f'{prefix}/episode_reward_min': np.min(all_rewards),
                f'{prefix}/episode_reward_max': np.max(all_rewards),
            }, step=self.global_frame)
            if best_reward is not None:
                arg_max_rewards = np.array(arg_max_rewards)
                wandb.log({
                    f'{prefix}/argmax_best_reward': np.mean(arg_max_rewards),
                    f'{prefix}/argmax_best_reward_std': np.std(arg_max_rewards),
                    f'{prefix}/argmax_best_reward_min': np.min(arg_max_rewards),
                    f'{prefix}/argmax_best_reward_max': np.max(arg_max_rewards),
                }, step=self.global_frame)
                x = arg_max_rewards
                y = all_rewards
                table = wandb.Table(
                    data=[[float(a), float(b)] for a, b in zip(x, y)],
                    columns=["arg_max_rewards", "episode_rewards"]
                )
                scatter = wandb.plot.scatter(
                    table,
                    x="arg_max_rewards",
                    y="episode_rewards",
                    title="ArgMax Rewards vs Episode Rewards"
                )
                wandb.log({f"{prefix}/reward_scatter": scatter}, step=self.global_frame)
            if goal is not None:
                plot_wandb_lines(raw_obs_distances, prefix, self.global_frame, key_name="obs_dist_ep", x_name="step", y_name="obs_dist")
            # latent distances
            plot_wandb_lines(latent_distances, prefix, self.global_frame, key_name="latent_dist_ep", x_name="step", y_name="latent_dist")
            plot_wandb_lines(task_reward, prefix, self.global_frame, key_name="task_reward_ep", x_name="step", y_name="task_reward")
            if len(ds_raw_reward) > 0:
                plot_wandb_lines(ds_raw_reward, prefix, self.global_frame, key_name="ds_task_reward_ep", x_name="step", y_name="task_reward")
                plot_wandb_lines(ds_latent_dist, prefix, self.global_frame, key_name="ds_latent_dist_ep", x_name="step", y_name="latent_dist")
            # Create combined rescaled plots using wandb.plot.line_series
            rescaled_latent = rescale_to_unit_range(latent_distances)
            rescaled_rewards = rescale_to_unit_range(task_reward)
            for ep in range(len(rescaled_latent)):
                if len(rescaled_latent[ep]) > 0 and len(rescaled_rewards[ep]) > 0:
                    assert len(rescaled_latent[ep]) == len(rescaled_rewards[ep])
                    xs = list(range(len(rescaled_latent[ep])))
                    ys = [
                        rescaled_latent[ep],
                        rescaled_rewards[ep],
                    ]
                    keys = ["latent_distance", "raw_reward"]
                    series = {keys[i]: ys[i] for i in range(len(keys))}
                    log_plotly_lines(f'{prefix}', ep, xs, series, self.global_frame)
                    
        if len(videos) > 0:
            video = record_video(f'{prefix}_TrajVideo_{self.global_frame}', videos, skip_frames=2)
            wandb.log({f'{prefix}/TrajVideo': video}, step=self.global_frame)


    @torch.no_grad()
    def potential_eval_extensive(self, replay_loader):
        """Goal-reaching evaluation with cosine-perturbed z. Runs in batches to cap parallel envs."""
        cosine_target = [0.99, 0.95, 0.9, 0.8, 0.5, 0.25, 0, -0.25, -0.5, -0.8, -0.9, -0.95, -1]
        custom_reward = self._make_custom_reward()
        sample_num = 10
        variant_sample_num = 10  # 5 variants per base actor
        resample_num = 1

        # Collect base actors and phi goals
        z_fit_actors: tp.List[torch.Tensor] = []
        z_fit_phi_goals: tp.List[torch.Tensor] = []
        z_argmax_actors: tp.List[torch.Tensor] = []
        z_argmax_phi_goals: tp.List[torch.Tensor] = []
        fit_variants: tp.Dict[str, tp.List[torch.Tensor]] = {str(target): [] for target in cosine_target}
        argmax_variants: tp.Dict[str, tp.List[torch.Tensor]] = {str(target): [] for target in cosine_target}

        fit_diags: tp.List[dict] = []
        for _ in range(sample_num):
            # fit sample
            meta, fit_diagnostics = _init_eval_meta(self, replay_loader, custom_reward)
            z_fit_actors.append(torch.as_tensor(meta['z'], device=self.cfg.device))
            z_fit_phi_goals.append(torch.as_tensor(meta['z_raw'], device=self.cfg.device))
            # record diagnostics per fit z
            try:
                if isinstance(fit_diagnostics, dict):
                    fit_diags.append(fit_diagnostics)
                else:
                    fit_diags.append({"raw": fit_diagnostics})
            except Exception:
                fit_diags.append({})
            # argmax sample
            goal_np, _, _, _ = self.get_argmax_goal_potential(replay_loader, custom_reward, return_indices=True)
            obs_zero = np.zeros_like(goal_np)
            meta = self.agent.get_goal_meta(goal_array=goal_np, obs_array=obs_zero)
            z_argmax_actors.append(torch.as_tensor(meta['z'], device=self.cfg.device))
            phi_raw = meta['z_raw']
            z_argmax_phi_goals.append(torch.as_tensor(phi_raw, device=self.cfg.device))

        # Pairwise cosine similarity within groups; compute now, log later with rewards
        fit_sim = None
        argmax_sim = None
        try:
            if len(z_fit_actors) > 0 and len(z_argmax_actors) > 0:
                def cosine_matrix(vectors: tp.List[torch.Tensor]) -> np.ndarray:
                    Z = torch.stack(vectors, dim=0)  # (N, D)
                    Z = Z.to(dtype=torch.float32)
                    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-8)
                    sim = (Z @ Z.t()).clamp(-1, 1)
                    return sim.detach().cpu().numpy()

                fit_sim = cosine_matrix(z_fit_actors)
                argmax_sim = cosine_matrix(z_argmax_actors)
        except Exception as e:
            logger.warning(f"Cosine matrix computation failed: {e}")

        # Create cosine variants (5 random variants per target, per base)
        for fit_phi_goal in z_fit_phi_goals:
            for target in cosine_target:
                for _ in range(variant_sample_num):
                    _, variant_z_actor = generate_cosine_variant(fit_phi_goal, target, device=self.cfg.device)
                    fit_variants[str(target)].append(variant_z_actor)
        for argmax_phi_goal in z_argmax_phi_goals:
            for target in cosine_target:
                for _ in range(variant_sample_num):
                    _, variant_z_actor = generate_cosine_variant(argmax_phi_goal, target, device=self.cfg.device)
                    argmax_variants[str(target)].append(variant_z_actor)

        # Assemble z vectors and index groups
        z_actors_summon: tp.List[torch.Tensor] = []
        z_actors_summon.extend(z_fit_actors)
        z_fit_indexes = np.arange(len(z_fit_actors))
        indexes_collections: tp.Dict[str, np.ndarray] = {"fit_raw": z_fit_indexes}
        for target in cosine_target:
            current_num = len(z_actors_summon)
            z_actors_summon.extend(fit_variants[str(target)])
            tmp_indexes = np.arange(current_num, len(z_actors_summon))
            indexes_collections[f"fit_{target}"] = tmp_indexes
        current_num = len(z_actors_summon)
        z_actors_summon.extend(z_argmax_actors)
        z_argmax_indexes = np.arange(len(z_argmax_actors)) + current_num
        indexes_collections["argmax_raw"] = z_argmax_indexes
        for target in cosine_target:
            current_num = len(z_actors_summon)
            z_actors_summon.extend(argmax_variants[str(target)])
            tmp_indexes = np.arange(current_num, len(z_actors_summon))
            indexes_collections[f"argmax_{target}"] = tmp_indexes

        z_actors_summon = torch.stack(z_actors_summon, dim=0).to(self.cfg.device)
        total_envs = z_actors_summon.shape[0]
        max_envs_per_batch = 400
        all_rewards = np.zeros((resample_num, total_envs,), dtype=np.float32)
        # Create a single vectorized env with max_envs_per_batch and reuse it across chunks
        eval_env = make_gym_async_vectorized(
            name=self.cfg.task,
            num_envs=max_envs_per_batch,
            obs_type=self.cfg.obs_type,
            frame_stack=self.cfg.frame_stack,
            action_repeat=self.cfg.action_repeat,
            seed=self.cfg.seed,
            image_wh=self.cfg.image_wh,
        )
        for i in range(resample_num):
            # Evaluate in chunks to limit memory usage
            for start in range(0, total_envs, max_envs_per_batch):
                end = min(start + max_envs_per_batch, total_envs)
                batch_z = z_actors_summon[start:end]
                batch_size = batch_z.shape[0]
                print(f"Initialize Environment, batch {start}:{end} (size={batch_size}), ep: {i} total: {total_envs}")
                obs, info = eval_env.reset()
                done = np.zeros((max_envs_per_batch,), dtype=bool)
                batch_rewards = np.zeros((max_envs_per_batch,), dtype=np.float32)
                # Rollout until all envs are terminated or truncated
                if batch_z.shape[0] < max_envs_per_batch:
                    z_placeholder = torch.zeros((max_envs_per_batch - batch_z.shape[0],) + batch_z.shape[1:], device=batch_z.device, dtype=batch_z.dtype)
                    batch_z = torch.cat([batch_z, z_placeholder], dim=0)
                while not np.all(done):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        actions = self.agent.act_inference(obs, batch_z)
                    obs, rew, term, trunc, infos = eval_env.step(actions)
                    batch_rewards += rew
                    done = term | trunc
                all_rewards[i, start:end] = batch_rewards[:end - start]
        all_rewards = np.mean(all_rewards, axis=0)  # (total_envs,)
        eval_env.close()
        print("Logging to wandb")
        # Log per-sample fit diagnostics as line charts
        try:
            if self.cfg.use_wandb and len(fit_diags) > 0:
                # Aggregate simple series across samples
                def _extract(key: str):
                    vals = []
                    for d in fit_diags:
                        v = d.get(key, np.nan)
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(np.nan)
                    return vals
                train_mse_series = _extract("train_mse")
                val_mse_series = _extract("val_mse")
                scale_series = []
                bias_series = []
                # handle possible alternative keys
                for d in fit_diags:
                    a = d.get("scale", d.get("a", np.nan))
                    b = d.get("bias", d.get("b", np.nan))
                    try:
                        scale_series.append(float(a))
                    except Exception:
                        scale_series.append(np.nan)
                    try:
                        bias_series.append(float(b))
                    except Exception:
                        bias_series.append(np.nan)

                x = list(range(len(fit_diags)))
                fig_diag = make_subplots(rows=2, cols=1, specs=[[{}], [{"secondary_y": True}]],
                                         subplot_titles=("fit diagnostics: MSE across samples",
                                                         "fit diagnostics: scale (left) & bias (right)"))
                # Top: MSEs
                if any([not np.isnan(v) for v in train_mse_series]):
                    fig_diag.add_trace(go.Scatter(x=x, y=train_mse_series, mode="lines+markers", name="train_mse"), row=1, col=1)
                if any([not np.isnan(v) for v in val_mse_series]):
                    fig_diag.add_trace(go.Scatter(x=x, y=val_mse_series, mode="lines+markers", name="val_mse"), row=1, col=1)
                fig_diag.update_yaxes(title_text="MSE", row=1, col=1)
                fig_diag.update_xaxes(title_text="sample idx", row=1, col=1)
                # Bottom: scale and bias
                if any([not np.isnan(v) for v in scale_series]):
                    fig_diag.add_trace(go.Scatter(x=x, y=scale_series, mode="lines+markers", name="scale(a)"), row=2, col=1, secondary_y=False)
                    fig_diag.update_yaxes(title_text="scale (a)", row=2, col=1, secondary_y=False)
                if any([not np.isnan(v) for v in bias_series]):
                    fig_diag.add_trace(go.Scatter(x=x, y=bias_series, mode="lines+markers", name="bias(b)"), row=2, col=1, secondary_y=True)
                    fig_diag.update_yaxes(title_text="bias (b)", row=2, col=1, secondary_y=True)
                fig_diag.update_xaxes(title_text="sample idx", row=2, col=1)
                fig_diag.update_layout(height=700, title_text="fit z diagnostics across samples")
                wandb.log({"extensive_plots/fit_diagnostics_lines": fig_diag}, step=self.global_frame)
        except Exception as e:
            logger.warning(f"Logging fit diagnostics failed: {e}")
        heatmap_colorscale = [[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']]
        # Combined figure: top row heatmaps, bottom row reward lines aligned with heatmaps order
        try:
            if self.cfg.use_wandb and fit_sim is not None and argmax_sim is not None:
                fig2 = make_subplots(
                    rows=2, cols=2,
                    specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                           [{"secondary_y": True}, {"secondary_y": True}]],
                    subplot_titles=(
                        f"z_fit_actors cosine (N={len(z_fit_actors)})",
                        f"z_argmax_actors cosine (N={len(z_argmax_actors)})",
                        "z_fit_actors rewards (left) & cosine to best (right)",
                        "z_argmax_actors rewards (left) & cosine to best (right)",
                    )
                )
                # Heatmaps (top row)
                fig2.add_trace(
                    go.Heatmap(z=fit_sim, zmin=-1, zmax=1, colorscale=heatmap_colorscale, colorbar=dict(title='cos')),
                    row=1, col=1
                )
                fig2.add_trace(
                    go.Heatmap(z=argmax_sim, zmin=-1, zmax=1, colorscale=heatmap_colorscale, showscale=False),
                    row=1, col=2
                )
                # Reward lines (bottom row), aligned to the order in heatmaps
                if "fit_raw" in indexes_collections:
                    fit_idxs = indexes_collections["fit_raw"]
                    fit_rewards = [float(x) for x in all_rewards[fit_idxs]]
                    # Add rewards on primary y-axis
                    fig2.add_trace(
                        go.Scatter(x=list(range(len(fit_rewards))), y=fit_rewards, mode="lines+markers", name="fit_rewards"),
                        row=2, col=1, secondary_y=False
                    )
                    # Cosine sim to best-reward z on secondary y-axis
                    try:
                        best_i = int(np.argmax(fit_rewards))
                        fit_cos_to_best = fit_sim[best_i].tolist()
                        fig2.add_trace(
                            go.Scatter(x=list(range(len(fit_cos_to_best))), y=fit_cos_to_best, mode="lines+markers", name="fit_cos_to_best"),
                            row=2, col=1, secondary_y=True
                        )
                        fig2.update_yaxes(title_text="episode reward", row=2, col=1, secondary_y=False)
                        fig2.update_yaxes(title_text="cosine sim", range=[-1, 1], row=2, col=1, secondary_y=True)
                    except Exception as e:
                        logger.warning(f"Plotting fit cosine-to-best failed: {e}")
                if "argmax_raw" in indexes_collections:
                    arg_idxs = indexes_collections["argmax_raw"]
                    arg_rewards = [float(x) for x in all_rewards[arg_idxs]]
                    # Add rewards on primary y-axis
                    fig2.add_trace(
                        go.Scatter(x=list(range(len(arg_rewards))), y=arg_rewards, mode="lines+markers", name="argmax_rewards"),
                        row=2, col=2, secondary_y=False
                    )
                    # Cosine sim to best-reward z on secondary y-axis
                    try:
                        best_j = int(np.argmax(arg_rewards))
                        argmax_cos_to_best = argmax_sim[best_j].tolist()
                        fig2.add_trace(
                            go.Scatter(x=list(range(len(argmax_cos_to_best))), y=argmax_cos_to_best, mode="lines+markers", name="argmax_cos_to_best"),
                            row=2, col=2, secondary_y=True
                        )
                        fig2.update_yaxes(title_text="episode reward", row=2, col=2, secondary_y=False)
                        fig2.update_yaxes(title_text="cosine sim", range=[-1, 1], row=2, col=2, secondary_y=True)
                    except Exception as e:
                        logger.warning(f"Plotting argmax cosine-to-best failed: {e}")
                fig2.update_layout(height=900, title_text="Cosine heatmaps and aligned rewards")
                wandb.log({"extensive_plots/z_group_cosine_and_rewards": fig2}, step=self.global_frame)
        except Exception as e:
            logger.warning(f"Logging combined cosine and rewards figure failed: {e}")

        # New: fit-argmx group (fit + argmax) heatmap and bottom dual-axis line (rewards & cosine to best)
        try:
            if self.cfg.use_wandb and len(z_fit_actors) > 0 and len(z_argmax_actors) > 0:
                # Build fit-argmx group vectors and rewards using only RAW groups (exclude cosine variants)
                big_vecs: tp.List[torch.Tensor] = []
                big_vecs.extend(z_fit_actors)
                big_vecs.extend(z_argmax_actors)

                def cosine_matrix_from_list(vectors: tp.List[torch.Tensor]) -> np.ndarray:
                    Z = torch.stack(vectors, dim=0).to(dtype=torch.float32)
                    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-8)
                    return (Z @ Z.t()).clamp(-1, 1).detach().cpu().numpy()

                big_sim = cosine_matrix_from_list(big_vecs)
                # Gather rewards in the same order as big_vecs
                if "fit_raw" in indexes_collections and "argmax_raw" in indexes_collections:
                    fit_idxs = indexes_collections["fit_raw"]
                    arg_idxs = indexes_collections["argmax_raw"]
                    big_rewards = np.concatenate([all_rewards[fit_idxs], all_rewards[arg_idxs]], axis=0)
                else:
                    # Fallback: infer by lengths
                    big_rewards = all_rewards[:len(big_vecs)]

                best_idx = int(np.argmax(big_rewards))
                cos_to_best = big_sim[best_idx].tolist()

                fig_big = make_subplots(
                    rows=2, cols=1,
                    specs=[[{"type": "heatmap"}], [{"secondary_y": True}]],
                    subplot_titles=(
                        f"fit-argmx group cosine (N={len(big_vecs)})",
                        "fit-argmx group rewards (left) & cosine to best (right)",
                    )
                )
                # Top heatmap with requested colorscale
                fig_big.add_trace(
                    go.Heatmap(z=big_sim, zmin=-1, zmax=1, colorscale=heatmap_colorscale, colorbar=dict(title='cos')),
                    row=1, col=1
                )
                # Bottom dual-axis line
                xs = list(range(len(big_rewards)))
                fig_big.add_trace(
                    go.Scatter(x=xs, y=[float(x) for x in big_rewards], mode="lines+markers", name="episode reward"),
                    row=2, col=1, secondary_y=False
                )
                fig_big.add_trace(
                    go.Scatter(x=xs, y=cos_to_best, mode="lines+markers", name="cosine to best"),
                    row=2, col=1, secondary_y=True
                )
                fig_big.update_yaxes(title_text="episode reward", row=2, col=1, secondary_y=False)
                fig_big.update_yaxes(title_text="cosine sim", range=[-1, 1], row=2, col=1, secondary_y=True)
                fig_big.update_layout(height=900, title_text="fit-argmx group: cosine heatmap and aligned rewards")
                wandb.log({"extensive_plots/fit-argmax_group_cosine_and_rewards": fig_big}, step=self.global_frame)
        except Exception as e:
            logger.warning(f"Logging fit-argmx group figure failed: {e}")
        # Log per-group stats (max, min, std); mean is visualized in the plot below
        # stats_log = {}
        # for key, idxs in indexes_collections.items():
        #     vals = all_rewards[idxs]
        #     stats_log[f"extensive/{key}_max"] = float(np.max(vals))
        #     stats_log[f"extensive/{key}_min"] = float(np.min(vals))
        #     stats_log[f"extensive/{key}_std"] = float(np.std(vals))
        # wandb.log(stats_log, step=self.global_frame)

        # Build box plots: per-cosine group distribution of episode rewards
        def build_box_data(prefix: str):
            data = []  # list of (label, values)
            xs_all = []
            raw_key = f"{prefix}_raw"
            if raw_key in indexes_collections:
                vals = all_rewards[indexes_collections[raw_key]]
                data.append(("1.0", vals.tolist()))
                xs_all.append(1.0)
            for ct in cosine_target:
                key = f"{prefix}_{ct}"
                if key in indexes_collections:
                    vals = all_rewards[indexes_collections[key]]
                    data.append((str(ct), vals.tolist()))
                    xs_all.append(float(ct))
            # Order categories by numeric cosine value
            order = np.argsort(np.array(xs_all)) if len(xs_all) > 0 else []
            if len(order) > 0:
                data = [data[i] for i in order]
            cats = [lbl for lbl, _ in data]
            return data, cats

        def log_box(name: str, data_and_cats):
            data, cats = data_and_cats
            if len(data) == 0:
                return
            fig = go.Figure()
            means = []
            for lbl, vals in data:
                means.append(float(np.mean(vals)))
                fig.add_trace(go.Box(
                    x=[lbl] * len(vals),
                    y=vals,
                    name=lbl,
                    boxmean=True,
                    boxpoints=False
                ))
            # Connect means with a line over categorical x-axis
            fig.add_trace(go.Scatter(
                x=cats,
                y=means,
                mode="lines+markers",
                name="mean",
                line=dict(color="black", width=2)
            ))
            fig.update_layout(
                title=f"{name}: cosine similarity vs episode reward (box + mean line)",
                xaxis_title="cosine similarity",
                yaxis_title="episode reward",
                xaxis=dict(categoryorder='array', categoryarray=cats)
            )
            wandb.log({f"extensive_plots/{name}_cos_vs_reward": fig}, step=self.global_frame)

        fit_data = build_box_data('fit')
        argmax_data = build_box_data('argmax')
        if len(fit_data[0]) > 0:
            print("Logging box fit")
            log_box("fit", fit_data)
        if len(argmax_data[0]) > 0:
            print("Logging box argmax")
            log_box("argmax", argmax_data)

        # Combined grouped box plot: fit vs argmax in one figure without interference
        try:
            if self.cfg.use_wandb:
                # Build maps for fast lookup
                fit_map = {lbl: vals for lbl, vals in fit_data[0]}
                arg_map = {lbl: vals for lbl, vals in argmax_data[0]}
                # Union categories and sort by numeric value
                cat_set = set(list(fit_map.keys()) + list(arg_map.keys()))
                if len(cat_set) > 0:
                    cats_sorted = sorted(cat_set, key=lambda s: float(s))
                    fig3 = go.Figure()
                    # Add side-by-side boxes using offsetgroup
                    for cat in cats_sorted:
                        if cat in fit_map:
                            fig3.add_trace(go.Box(
                                x=[cat] * len(fit_map[cat]),
                                y=fit_map[cat],
                                name=f"fit {cat}",
                                legendgroup="fit",
                                offsetgroup="fit",
                                marker_color="#1f77b4",
                                boxmean=True,
                                boxpoints=False,
                                showlegend=False  # reduce legend clutter per-category
                            ))
                        if cat in arg_map:
                            fig3.add_trace(go.Box(
                                x=[cat] * len(arg_map[cat]),
                                y=arg_map[cat],
                                name=f"argmax {cat}",
                                legendgroup="argmax",
                                offsetgroup="argmax",
                                marker_color="#ff7f0e",
                                boxmean=True,
                                boxpoints=False,
                                showlegend=False
                            ))
                    # Mean trend lines for both groups
                    fit_means_x, fit_means_y = [], []
                    arg_means_x, arg_means_y = [], []
                    for cat in cats_sorted:
                        if cat in fit_map and len(fit_map[cat]) > 0:
                            fit_means_x.append(cat)
                            fit_means_y.append(float(np.mean(fit_map[cat])))
                        if cat in arg_map and len(arg_map[cat]) > 0:
                            arg_means_x.append(cat)
                            arg_means_y.append(float(np.mean(arg_map[cat])))
                    if len(fit_means_x) > 0:
                        fig3.add_trace(go.Scatter(
                            x=fit_means_x, y=fit_means_y, mode="lines+markers",
                            name="fit mean", legendgroup="fit", line=dict(color="#1f77b4", width=2)
                        ))
                    if len(arg_means_x) > 0:
                        fig3.add_trace(go.Scatter(
                            x=arg_means_x, y=arg_means_y, mode="lines+markers",
                            name="argmax mean", legendgroup="argmax", line=dict(color="#ff7f0e", width=2)
                        ))
                    fig3.update_layout(
                        title="fit vs argmax: cosine similarity vs episode reward (grouped boxes + mean lines)",
                        xaxis_title="cosine similarity",
                        yaxis_title="episode reward",
                        xaxis=dict(categoryorder='array', categoryarray=cats_sorted),
                        boxmode='group'
                    )
                    wandb.log({"extensive_plots/fit_argmax_cos_vs_reward_grouped_box": fig3}, step=self.global_frame)
        except Exception as e:
            logger.warning(f"Logging grouped box figure failed: {e}")


    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        # this is just a dumb security check to not forget about it
        payload = {}
        for k in self._CHECKPOINTED_KEYS:
            if k not in exclude:
                if k in self.__dict__:
                    payload[k] = self.__dict__[k]
                else:
                    print("Warning: %s not found in __dict__" % k)
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)
        # remove older checkpoints that is older than 20 checkpoints
        models_dir = os.path.join(self.work_dir, "models")
        if os.path.exists(models_dir):
            models = os.listdir(models_dir)
            models.sort(key=lambda x: int(x.split('.')[0]))
            for model in models[:-20]:
                os.remove(os.path.join(models_dir, model))

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = (), num_episodes=None, use_pixels=False) -> None:
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
                if hasattr(self, name):
                    setattr(self, name, val)
                else:
                    print("Warning: %s not found in self" % name)

    def finalize(self) -> None:
        print("Running final test", flush=True)

        domain_tasks = {
            "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
            "quadruped": ['stand', 'walk', 'run', 'jump'],
            "walker": ['stand', 'walk', 'run', 'flip'],
            "jaco": ['reach_top_left', 'reach_top_right', 'reach_bottom_left', 'reach_bottom_right'],
        }
        if self.domain not in domain_tasks:
            return
        eval_hist = self.eval_rewards_history
        rewards = {}
        videos = {}
        infos = {}
        for name in domain_tasks[self.domain]:
            task = "_".join([self.domain, name])
            self.cfg.task = task
            self.cfg.custom_reward = task  # for the replay buffer
            self.cfg.seed += 10000000  # for the sake of avoiding similar seeds
            self.eval_env = self._make_env()
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = self.cfg.num_final_eval_episodes
            info, video = self.eval(final_eval=True)
            rewards[name] = self.eval_rewards_history.copy()
            infos[name] = info
            videos[name] = video
            # try:
            #     diff_info, diff_video = self.eval(final_eval=True, prefix='diff')
            #     videos[name + '_diff'] = diff_video
            #     infos[name + '_diff'] = diff_info
            #     rewards[name + '_diff'] = self.eval_rewards_history.copy()
            # except Exception as e:
            #     print(f"Error evaluating {name} diff: {e}")
        # with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
        #     for name in domain_tasks[self.domain]:
        #         video = record_video(f'Final_{name}', videos[name], skip_frames=2)
        #         # diff_video = record_video(f'Final_{name}_diff', videos[name + '_diff'], skip_frames=2)
        #         wandb.log({f'Final_{name}': video}, step=self.global_frame)
        #         # wandb.log({f'Final_{name}_diff': diff_video}, step=self.global_frame)
        #         for k, v in infos[name].items():
        #             log(f'final/{name}/{k}', v)
        #         # for k, v in diff_info.items():
        #         #     log(f'final/{name}_diff/{k}', v)
        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)
    
    def init_phi_dataset(self):
        if not hasattr(self, 'phi_dataset'):       
            obs_horizon = self.cfg.agent.obs_horizon if hasattr(self.cfg.agent, 'obs_horizon') else 1
            self.phi_dataset = PhiWalkerDataset(
                data_dir=os.path.join(self.work_dir, "phi_dataset"),
                obs_horizon=obs_horizon,
                total_episodes=self.cfg.phi_total_episodes,
                discount=self.cfg.discount,
                goal_future=self.cfg.future,
                p_randomgoal=self.cfg.p_randomgoal,
            )

    def train_phi(self, is_init: bool = False):
        print("Training Phi")
        self.prepare_phi_dataset(is_init)
        # metrics_summon = defaultdict(list)
        train_steps = self.cfg.phi_net_pretrain_steps
        pbar = trange(train_steps, desc="Training Phi", position=1, leave=True)
        self.agent.feature_learner.train()
        for batch in self.phi_dataloader:
            metrics = self.agent.update_phi(batch)
            self.update_train_metrics(metrics)
            self.phi_step += 1
            self.global_step += 1
            pbar.update(1)
            pbar.set_description(f"Local, Phi Loss: {metrics['hilp/value_loss']:.4f}")
            self.global_progress_bar.update(1)
            self.global_progress_bar.set_description(f"Global,Training Phi")
            if self.phi_step % train_steps == 0:
                break
        _checkpoint_filepath = os.path.join(self.work_dir, f"phi_pretrained_tmp.pt")
        self.save_checkpoint(_checkpoint_filepath)
        del self.phi_dataloader

    def prepare_phi_dataset(self, is_init=False):
        self.init_phi_dataset()
        total_num = self.cfg.phi_rollout_num
        start_time = time.time()
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
            "drop_last": True,
        }
        self.phi_dataloader = InfiniteDataLoaderWrapper(self.phi_dataset, dataloader_cfg)
        end_time = time.time()
        if self.cfg.use_wandb:
            wandb.log({f"train/phi_dataset_time": end_time - start_time, 'phi_dataset_ep_num': self.phi_dataset.meta_info['current_size']}, step=self.global_step)
        else:
            print(f"train/phi_dataset_time: {end_time - start_time}")

    def _collect_episodes(self, random_sample=True, phi_obs=None, phi_future_obs=None, is_init=False):
        """Collect episodes using gymnasium AsyncVectorEnv.

        Returns:
            episodes: List[dict], one per env, with keys:
                - data: { 'observation', 'actions', 'rewards', 'discount' }
                - meta: {}
            mean_len: float, average episode length over envs
        """
        obs, infos = self.train_env.reset()
        physics = infos['physics']
        action_shape = self.train_env.action_space.shape

        data_buffer = {
            "observation": np.empty((self.cfg.num_train_envs, self.cfg.num_episode_steps, *obs.shape[1:]), dtype=obs.dtype),
            "action": np.empty((self.cfg.num_train_envs, self.cfg.num_episode_steps, *action_shape[1:]), dtype=np.float32),
            "reward": np.empty((self.cfg.num_train_envs, self.cfg.num_episode_steps), dtype=np.float32),
            'physics': np.empty((self.cfg.num_train_envs, self.cfg.num_episode_steps, *physics.shape[1:]), dtype=physics.dtype),
        }
        t = 0
        if self.cfg.agent.name == 'sf':
            z = self.agent.sample_z(size=self.cfg.num_train_envs)
        else:
            z = self.agent.sample_z(obs=phi_obs, future_obs=phi_future_obs, size=self.cfg.num_train_envs, random_sample=random_sample)
            if z.shape[0] < self.cfg.num_train_envs:
                delta_num = self.cfg.num_train_envs - z.shape[0]
                d = z.shape[1]
                z_r = torch.randn((delta_num, d), dtype=torch.float32, device=z.device)
                z_r = math.sqrt(d) * F.normalize(z_r, dim=1)
                z = torch.cat([z, z_r], axis=0)

        while t < self.cfg.num_episode_steps:
            # Random actions by default; integrate policy here if desired
            # actions = np.random.uniform(low=act_low, high=act_high, size=(num_envs,) + act_shape).astype(np.float32)
            data_buffer['observation'][:, t] = obs
            data_buffer['physics'][:, t] = physics
            if is_init:
                actions = self.train_env.action_space.sample()
            else:
                actions = self.agent.act_inference(obs, z)
            obs, rew, term, trunc, infos = self.train_env.step(actions)
            # dones = np.logical_or(term, trunc)
            data_buffer['action'][:, t] = actions
            data_buffer['reward'][:, t] = rew
            physics = infos['physics']
            t += 1
        return data_buffer

    def update_train_metrics(self, metrics_summon: tp.Dict[str, float]) -> None:
        for k, v in metrics_summon.items():
            metrics_summon[k] = np.mean(v)
        if self.cfg.use_wandb:
            wandb.log({f"train/{'_'.join(k.split('/'))}" if "/" in k else f"train/{k}": v for k, v in metrics_summon.items()}, step=self.global_step)
        else:
            for k, v in metrics_summon.items():
                print(f"train/{'_'.join(k.split('/'))}" if "/" in k else f"train/{k}: {v}")

    def train_policy(self):
        print("Training Policy")
        # sac training
        loader_cfg = {
            "batch_size": self.cfg.num_train_envs,
            "shuffle": True,
            "num_workers": self.cfg.num_workers,
            "drop_last": True,
        }
        if hasattr(self, "phi_dataset"):
            policy_phi_dataloader = InfiniteDataLoaderWrapper(self.phi_dataset, loader_cfg)
        else:
            self.prepare_phi_dataset()
            policy_phi_dataloader = InfiniteDataLoaderWrapper(self.phi_dataset, loader_cfg)
        for batch in policy_phi_dataloader:
            phi_obs = batch['obs'].to(self.cfg.device)
            phi_future_obs = batch['future_obs'].to(self.cfg.device)
            episodes = self._collect_episodes(random_sample=False, phi_obs=phi_obs, phi_future_obs=phi_future_obs)
            self.replay_loader.add_episodes(episodes, self.cfg.num_train_envs)
            if len(self.replay_loader) > self.cfg.replay_buffer_init_size:
                for _ in range(self.cfg.sac_optim_steps):
                    sac_batch = self.replay_loader.sample(self.cfg.batch_size)
                    metrics = self.agent.update_batch(sac_batch)
                    self.policy_step += 1
                    self.global_step += 1
                    self.global_progress_bar.update(1)
                    self.global_progress_bar.set_description(f"Global, SAC Training")
                    metrics['replay_loader_steps'] = len(self.replay_loader)
                    self.update_train_metrics(metrics)
                    if self.policy_step % self.cfg.save_every_steps == 0:
                        _checkpoint_filepath = os.path.join(self.work_dir, "models", f"{self.global_step}.pt")
                        self.save_checkpoint(_checkpoint_filepath)
                    if self.policy_step % self.cfg.rollout_every_steps == 0:
                        policy_phi_dataloader.close_loader()
                        return
        return 

@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == '__main__':
    main()