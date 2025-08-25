import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import queue

import numpy as np
import zarr

# ensure local repo is importable
sys.path.append(os.getcwd())
from tqdm import tqdm
from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
from legged_gym.dataset.replay_buffer import ReplayBuffer
import torch
from isaacgym.torch_utils import torch_rand_float


class ReplayBufferWriter:
    def __init__(self, zarr_path: str, traj_type: str, stop_event: threading.Event, max_queue_size: int = 8):
        self.zarr_path = zarr_path
        self.traj_type = traj_type  # "constant" or "switch"
        self.stop_event = stop_event
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._run, name=f"ReplayWriter-{traj_type}", daemon=True)
        self._started = False

    def start(self):
        if not self._started:
            self._started = True
            self.thread.start()

    def join(self):
        if not self._started:
            return
        # drain remaining items
        self.queue.join()

    def put_batch(self, batch: Dict[str, Any]):
        # blocks if queue full
        self.queue.put(batch)

    def _open_or_repair(self) -> ReplayBuffer:
        try:
            return ReplayBuffer.create_from_path(self.zarr_path, mode="a")
        except AssertionError as e:
            print(f"[ReplayBufferWriter-{self.traj_type}] detected mismatched lengths, attempting repair: {e}")
            group = zarr.open(self.zarr_path, mode="a")
            meta = group["meta"]
            if "episode_ends" not in meta or meta["episode_ends"].size == 0:
                # empty meta, reset dataset
                del group["data"]
                del group["meta"]
                return ReplayBuffer.create_from_path(self.zarr_path, mode="a")
            target = int(meta["episode_ends"][-1])
            data_grp = group.get("data", None)
            if data_grp is not None:
                for key, arr in data_grp.items():
                    if arr.shape[0] != target:
                        print(f"[Repair] Truncating {key} from {arr.shape[0]} to {target}")
                        arr.resize((target,) + arr.shape[1:])
            # done, reopen
            return ReplayBuffer.create_from_path(self.zarr_path, mode="a")

    def _run(self):
        # Open or create zarr replay buffer (with repair if needed)
        buffer = self._open_or_repair()
        while True:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            try:
                episodes: List[Dict[str, Any]] = item["episodes"]
                t0 = time.time()
                # Build bulk arrays by concatenating along time dimension
                lengths = np.array([len(ep["data"]["obs"]) for ep in episodes], dtype=np.int64)
                total_T = int(lengths.sum())

                # Collect and concat keys present
                data_keys = ["obs", "critic_obs", "actions", "rewards", "dones", "root_states", "commands"]
                concat_data: Dict[str, np.ndarray] = {}
                for key in data_keys:
                    arrays = []
                    for ep in episodes:
                        arr = ep["data"].get(key, None)
                        if arr is None:
                            arrays = []
                            break
                        if not isinstance(arr, np.ndarray):
                            arr = np.asarray(arr)
                        arrays.append(arr)
                    if arrays:
                        concat_data[key] = np.concatenate(arrays, axis=0)

                # Append all step data in one call
                if concat_data:
                    buffer.add_chunked_data(concat_data)

                # Compute episode_ends based on current steps
                curr_steps = buffer.n_steps
                start_steps = curr_steps
                episode_ends = start_steps + np.cumsum(lengths)

                # Build per-episode meta
                rewards = np.array([float(ep["meta"].get("reward", 0.0)) for ep in episodes], dtype=np.float32)
                if self.traj_type == "switch":
                    cmd_A = np.stack([np.asarray(ep["meta"]["command_A"], dtype=np.float32) for ep in episodes])
                    cmd_B = np.stack([np.asarray(ep["meta"].get("command_B", np.zeros_like(cmd_A[0])), dtype=np.float32) for ep in episodes])
                    meta_bulk = {
                        "episode_ends": episode_ends.astype(np.int64),
                        "episode_reward": rewards,
                        "episode_command_A": cmd_A,
                        "episode_command_B": cmd_B,
                    }
                else:
                    cmd = np.stack([np.asarray(ep["meta"]["command"], dtype=np.float32) for ep in episodes])
                    meta_bulk = {
                        "episode_ends": episode_ends.astype(np.int64),
                        "episode_reward": rewards,
                        "episode_command": cmd,
                    }
                buffer.add_chunked_meta(meta_bulk)

                dt = time.time() - t0
                print(f"[ReplayBufferWriter-{self.traj_type}] wrote {len(episodes)} episodes (T={total_T}) in {dt:.3f}s (total_eps={buffer.n_episodes})")
            finally:
                self.queue.task_done()
        # flush done


class TrajectoryDataCollector:
    def __init__(
        self,
        task_name: str = "h1int",
        num_envs: int = 4,
        headless: bool = True,
        load_checkpoint: Optional[str] = None,
        output_root: str = "collected_trajectories_v2",
        flush_episodes: int = 4096,
        max_learning_iter: int = 40000,
    ) -> None:
        self.task_name = task_name
        self.num_envs = num_envs
        self.headless = headless
        self.load_checkpoint = load_checkpoint
        self.output_root = output_root
        self.flush_episodes = flush_episodes
        self.max_learning_iter = max_learning_iter

        # Saving via ReplayBuffer writers (two buffers: constant and switch)
        os.makedirs(self.output_root, exist_ok=True)
        self.constant_rb_path = os.path.join(self.output_root, "constant.zarr")
        self.switch_rb_path = os.path.join(self.output_root, "switch.zarr")
        self._stop_event = threading.Event()
        # queue size small to apply backpressure
        self.constant_writer = ReplayBufferWriter(self.constant_rb_path, "constant", self._stop_event, max_queue_size=8)
        self.switch_writer = ReplayBufferWriter(self.switch_rb_path, "switch", self._stop_event, max_queue_size=8)
        self.constant_writer.start()
        self.switch_writer.start()

        # per-type accumulation before flush
        self._accum_constant: List[Dict[str, Any]] = []
        self._accum_switch: List[Dict[str, Any]] = []

        # create env with 10s episode
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=self.task_name)
        self.env_cfg.env.num_envs = self.num_envs
        self.env_cfg.env.episode_length_s = 10.0

        # prevent in-episode command resampling; we will control commands manually
        self.env_cfg.commands.resampling_time = 10.0  # still equals episode length; first step won't hit modulo

        # build args and env
        from isaacgym import gymutil
        args = gymutil.parse_arguments(description="Trajectory Data Collector", custom_parameters=[
            {"name": "--task", "type": str, "default": self.task_name},
            {"name": "--num_envs", "type": int, "default": self.num_envs},
            {"name": "--headless", "action": "store_true", "default": self.headless},
            # allow extra CLI flags so gymutil doesn't error out
            {"name": "--num_constant", "type": int, "default": 0},
            {"name": "--num_switch", "type": int, "default": 0},
            {"name": "--load_checkpoint", "type": str},
            {"name": "--output_root", "type": str},
            {"name": "--flush_episodes", "type": int, "default": self.flush_episodes},
        ])
        args.headless = self.headless or getattr(args, "headless", False)
        args.num_envs = self.num_envs
        self.flush_episodes = getattr(args, "flush_episodes", self.flush_episodes)

        self.env, _ = task_registry.make_env(name=self.task_name, args=args, env_cfg=self.env_cfg)
        self.device = self.env.device

        # optional policy for actions
        self.policy = None
        if self.load_checkpoint:
            try:
                self.train_cfg.runner.resume = True
                self.train_cfg.runner.resume_path = self.load_checkpoint
                ppo_runner, _ = task_registry.make_alg_runner(env=self.env, name=self.task_name, args=args, train_cfg=self.train_cfg, log_root=None)
                self.policy = ppo_runner.get_inference_policy(device=self.device)
            except Exception:
                self.policy = None

        # reset once to init buffers
        _, _ = self.env.reset()

    def _wait_for_queue(self, which: str) -> None:
        # backpressure: block when queue is too full (>= num_envs)
        q = self.constant_writer.queue if which == "constant" else self.switch_writer.queue
        while q.qsize() >= self.num_envs:
            time.sleep(0.01)

    def _get_actions(self) -> torch.Tensor:
        if self.policy is not None:
            with torch.inference_mode():
                obs = self.env.get_observations()
                critic_obs = self.env.get_privileged_observations()
                actions, _ = self.policy.act_inference(obs, privileged_obs=critic_obs)
                return actions
        return torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample commands for specified environments.
        Prefer the environment's curriculum-based sampler if enabled; otherwise
        sample uniformly from configured ranges.
        """
        cfg_ranges = self.env.cfg.commands.ranges
        use_curriculum = getattr(self.env.cfg.commands, "curriculum", False)

        if use_curriculum:
            # Use curriculum grid to sample vx/yaw
            self.env.update_command_curriculum_grid(env_ids)

            # Fill remaining command dimensions (3..9)
            gait_freq = torch_rand_float(cfg_ranges.gait_frequency[0], cfg_ranges.gait_frequency[1], (len(env_ids), 1), device=self.device).squeeze(1)
            phases = torch.tensor([0.0, 0.5], device=self.device)
            rand_idx = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
            phase = phases[rand_idx]
            swing_height = torch_rand_float(cfg_ranges.foot_swing_height[0], cfg_ranges.foot_swing_height[1], (len(env_ids), 1), device=self.device).squeeze(1)

            # Body pose parameters
            if self.env.cfg.env.observe_body_height:
                body_height = torch_rand_float(cfg_ranges.body_height[0], cfg_ranges.body_height[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_height = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_body_pitch:
                body_pitch = torch_rand_float(cfg_ranges.body_pitch[0], cfg_ranges.body_pitch[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_pitch = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_waist_roll:
                waist_roll = torch_rand_float(cfg_ranges.waist_roll[0], cfg_ranges.waist_roll[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                waist_roll = torch.zeros(len(env_ids), device=self.device)

            # Assign to env commands
            self.env.commands[env_ids, 3] = gait_freq
            self.env.commands[env_ids, 4] = phase
            self.env.commands[env_ids, 5] = 0.5  # duration fixed as in env
            self.env.commands[env_ids, 6] = swing_height
            self.env.commands[env_ids, 7] = body_height
            if self.env.cfg.env.observe_body_pitch:
                self.env.commands[env_ids, 8] = body_pitch
            if self.env.cfg.env.observe_waist_roll:
                self.env.commands[env_ids, 9] = waist_roll

            # Apply high-speed constraints similar to env
            velocity_level = torch.clip(1.0*torch.norm(self.env.commands[env_ids, :2], dim=-1) + 0.5*torch.abs(self.env.commands[env_ids, 2]), min=1)
            high_speed_mask = velocity_level > 1.8
            if high_speed_mask.any():
                self.env.commands[env_ids[high_speed_mask], 3] = torch.clip(self.env.commands[env_ids[high_speed_mask], 3], min=2.0)
                if self.env.cfg.env.observe_body_pitch:
                    self.env.commands[env_ids[high_speed_mask], 8] = torch.clip(self.env.commands[env_ids[high_speed_mask], 8], max=0.3)
                if self.env.cfg.env.observe_waist_roll:
                    self.env.commands[env_ids[high_speed_mask], 9] = torch.clip(self.env.commands[env_ids[high_speed_mask], 9], min=-0.15, max=0.15)

            # Low body height constraints
            if self.env.cfg.env.observe_body_height:
                low_height_mask = self.env.commands[env_ids, 7] < -0.15
                if low_height_mask.any():
                    self.env.commands[env_ids[low_height_mask], 6] = torch.clip(self.env.commands[env_ids[low_height_mask], 6], max=0.20)
                if self.env.cfg.env.observe_body_pitch:
                    lower_height_mask = self.env.commands[env_ids, 7] < -0.2
                    if lower_height_mask.any():
                        self.env.commands[env_ids[lower_height_mask], 8] = torch.clip(self.env.commands[env_ids[lower_height_mask], 8], max=0.3)
        else:
            # Fallback: sample uniformly from configured static ranges (no curriculum)
            lin_vel_x = torch_rand_float(cfg_ranges.lin_vel_x[0], cfg_ranges.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
            lin_vel_y = torch_rand_float(cfg_ranges.lin_vel_y[0], cfg_ranges.lin_vel_y[1], (len(env_ids), 1), device=self.device).squeeze(1)
            ang_vel_yaw = torch_rand_float(cfg_ranges.ang_vel_yaw[0], cfg_ranges.ang_vel_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)

            gait_freq = torch_rand_float(cfg_ranges.gait_frequency[0], cfg_ranges.gait_frequency[1], (len(env_ids), 1), device=self.device).squeeze(1)
            phases = torch.tensor([0.0, 0.5], device=self.device)
            rand_idx = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
            phase = phases[rand_idx]
            swing_height = torch_rand_float(cfg_ranges.foot_swing_height[0], cfg_ranges.foot_swing_height[1], (len(env_ids), 1), device=self.device).squeeze(1)

            if self.env.cfg.env.observe_body_height:
                body_height = torch_rand_float(cfg_ranges.body_height[0], cfg_ranges.body_height[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_height = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_body_pitch:
                body_pitch = torch_rand_float(cfg_ranges.body_pitch[0], cfg_ranges.body_pitch[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                body_pitch = torch.zeros(len(env_ids), device=self.device)
            if self.env.cfg.env.observe_waist_roll:
                waist_roll = torch_rand_float(cfg_ranges.waist_roll[0], cfg_ranges.waist_roll[1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                waist_roll = torch.zeros(len(env_ids), device=self.device)

            self.env.commands[env_ids, 0] = lin_vel_x
            self.env.commands[env_ids, 1] = lin_vel_y
            self.env.commands[env_ids, 2] = ang_vel_yaw
            self.env.commands[env_ids, 3] = gait_freq
            self.env.commands[env_ids, 4] = phase
            self.env.commands[env_ids, 5] = 0.5
            self.env.commands[env_ids, 6] = swing_height
            self.env.commands[env_ids, 7] = body_height
            if self.env.cfg.env.observe_body_pitch:
                self.env.commands[env_ids, 8] = body_pitch
            if self.env.cfg.env.observe_waist_roll:
                self.env.commands[env_ids, 9] = waist_roll

    def _collect_episode(self, trajectory_type: str) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Collect a single episode for all environments and return episodes list"""
        # Reset environment
        obs, critic_obs = self.env.reset()
        
        # Sample initial commands for all envs
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        self._resample_commands(env_ids)
        
        # Always store the initial commands as cmd_A
        cmd_A = self.env.commands.clone()
        
        # For switch trajectories, prepare second command set
        if trajectory_type == "switch":
            # Sample second command set
            self._resample_commands(env_ids)
            cmd_B = self.env.commands.clone()
            # Reset to first command set
            self.env.commands = cmd_A.clone()
        else:
            # For constant trajectories, cmd_B is not used
            cmd_B = None
        
        # Initialize buffers
        obs_buffers = [[] for _ in range(self.env.num_envs)]
        critic_obs_buffers = [[] for _ in range(self.env.num_envs)]
        act_buffers = [[] for _ in range(self.env.num_envs)]
        rew_buffers = [[] for _ in range(self.env.num_envs)]
        done_buffers = [[] for _ in range(self.env.num_envs)]
        root_buffers = [[] for _ in range(self.env.num_envs)]
        cmd_buffers = [[] for _ in range(self.env.num_envs)]
        
        # Track active environments
        active_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        
        # Episode loop
        t = 0
        max_steps = int(self.env_cfg.env.episode_length_s / self.env.dt)
        
        while t < max_steps and active_mask.any():
            # Get actions
            actions = self._get_actions()
            
            # Step environment
            obs, critic_obs, step_rewards, dones, infos = self.env.step(actions)
            
            # Switch commands halfway through for switch trajectories
            if trajectory_type == "switch" and t == max_steps // 2:
                self.env.commands = cmd_B.clone()
            
            # Record data for active environments
            for eid in range(self.env.num_envs):
                if not bool(active_mask[eid]):
                    continue
                obs_buffers[eid].append(obs[eid].detach().cpu().numpy())
                critic_obs_buffers[eid].append(critic_obs[eid].detach().cpu().numpy())
                act_buffers[eid].append(actions[eid].detach().cpu().numpy())
                rew_buffers[eid].append(float(step_rewards[eid].item()))
                done_buffers[eid].append(bool(dones[eid].item()))
                root_buffers[eid].append(self.env.root_states[eid].detach().cpu().numpy())
                cmd_buffers[eid].append(self.env.commands[eid].detach().cpu().numpy())

            t += 1
            # inactivate envs that are done at this step
            if dones.any():
                active_mask[dones > 0] = False
            # break when all envs finished early
            if (~active_mask).all():
                break

        # finalize: build episodes
        episodes: List[Dict[str, Any]] = []
        saved_rewards: List[float] = []

        for eid in range(self.env.num_envs):
            traj_rew = float(sum(rew_buffers[eid]))
            saved_rewards.append(traj_rew)

            data = {
                "obs": np.array(obs_buffers[eid]),
                "critic_obs": np.array(critic_obs_buffers[eid]),
                "actions": np.array(act_buffers[eid]),
                "rewards": np.array(rew_buffers[eid], dtype=np.float32),
                "dones": np.array(done_buffers[eid], dtype=bool),
                "root_states": np.array(root_buffers[eid]),
                "commands": np.array(cmd_buffers[eid]),
            }

            meta_entry: Dict[str, Any] = {
                "reward": traj_rew,
            }
            if trajectory_type == "switch":
                meta_entry["command_A"] = cmd_A[eid].detach().cpu().numpy().tolist()
                meta_entry["command_B"] = cmd_B[eid].detach().cpu().numpy().tolist() if cmd_B is not None else None
            else:
                meta_entry["command"] = cmd_A[eid].detach().cpu().numpy().tolist()

            episodes.append({"data": data, "meta": meta_entry})

        # allow curriculum tick
        self.env.training_curriculum(num_steps=self.curriculum_factor)
        return episodes, saved_rewards

    def _flush_if_needed(self, which: str) -> None:
        if which == "constant" and len(self._accum_constant) >= self.flush_episodes:
            # print queue size before enqueue
            print(f"[Collector] constant queue size before put: {self.constant_writer.queue.qsize()} (accum={len(self._accum_constant)})")
            self._wait_for_queue("constant")
            batch = {"episodes": self._accum_constant[:self.flush_episodes]}
            self.constant_writer.put_batch(batch)
            print(f"[Collector] constant queue size after put: {self.constant_writer.queue.qsize()}")
            self._accum_constant = self._accum_constant[self.flush_episodes:]
        elif which == "switch" and len(self._accum_switch) >= self.flush_episodes:
            print(f"[Collector] switch queue size before put: {self.switch_writer.queue.qsize()} (accum={len(self._accum_switch)})")
            self._wait_for_queue("switch")
            batch = {"episodes": self._accum_switch[:self.flush_episodes]}
            self.switch_writer.put_batch(batch)
            print(f"[Collector] switch queue size after put: {self.switch_writer.queue.qsize()}")
            self._accum_switch = self._accum_switch[self.flush_episodes:]

    def _final_flush(self) -> None:
        if len(self._accum_constant) > 0:
            print(f"[Collector] final flush constant (accum={len(self._accum_constant)}) queue={self.constant_writer.queue.qsize()}")
            self._wait_for_queue("constant")
            self.constant_writer.put_batch({"episodes": self._accum_constant})
            self._accum_constant = []
        if len(self._accum_switch) > 0:
            print(f"[Collector] final flush switch (accum={len(self._accum_switch)}) queue={self.switch_writer.queue.qsize()}")
            self._wait_for_queue("switch")
            self.switch_writer.put_batch({"episodes": self._accum_switch})
            self._accum_switch = []
        # wait writers
        self.constant_writer.join()
        self.switch_writer.join()

    def collect(self, num_constant: int, num_switch: int) -> None:
        max_learning_iter = self.max_learning_iter
        total_eps_num = (num_constant + num_switch) * self.num_envs
        self.curriculum_factor = max_learning_iter / total_eps_num
        try:
            produced_const = 0
            produced_switch = 0
            step = 0
            while produced_const < num_constant or produced_switch < num_switch:
                # decide which type to collect next to balance distribution across curriculum
                if produced_const >= num_constant:
                    next_type = "switch"
                elif produced_switch >= num_switch:
                    next_type = "constant"
                else:
                    # proportional fairness: pick the type that is behind in its quota
                    # compare produced_const/num_constant vs produced_switch/num_switch without floats
                    if produced_const * num_switch <= produced_switch * num_constant:
                        next_type = "constant"
                    else:
                        next_type = "switch"

                if next_type == "constant":
                    print(f"[Collector] step {step} collecting constant ({produced_const+1}/{num_constant}) | q_const={self.constant_writer.queue.qsize()} q_switch={self.switch_writer.queue.qsize()}")
                    episodes, _ = self._collect_episode("constant")
                    self._accum_constant.extend(episodes)
                    self._flush_if_needed("constant")
                    produced_const += 1
                else:
                    print(f"[Collector] step {step} collecting switch ({produced_switch+1}/{num_switch}) | q_const={self.constant_writer.queue.qsize()} q_switch={self.switch_writer.queue.qsize()}")
                    episodes, _ = self._collect_episode("switch")
                    self._accum_switch.extend(episodes)
                    self._flush_if_needed("switch")
                    produced_switch += 1

                step += 1
        finally:
            # Ensure all pending batches complete before shutdown
            self._final_flush()
            self._stop_event.set()

    def __del__(self):
        # signal stop and flush
        try:
            self._stop_event.set()
            self._final_flush()
        except Exception:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trajectory data collector (runner-style)")
    parser.add_argument("--task", type=str, default="h1int")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="collected_trajectories_v2")
    parser.add_argument("--num_constant", type=int, default=10)
    parser.add_argument("--num_switch", type=int, default=10)
    parser.add_argument("--flush_episodes", type=int, default=4096)
    args = parser.parse_args()

    collector = TrajectoryDataCollector(
        task_name=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        load_checkpoint=args.load_checkpoint,
        output_root=args.output_root,
        flush_episodes=args.flush_episodes,
        max_learning_iter=40000,
    )
    collector.collect(args.num_constant, args.num_switch)


if __name__ == "__main__":
    main()


