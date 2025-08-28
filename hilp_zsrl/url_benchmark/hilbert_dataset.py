#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import zarr

from url_benchmark.replay_buffer import ReplayBuffer
from url_benchmark.in_memory_replay_buffer import EpisodeBatch


class HilbertRepresentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 types: Optional[List[str]] = None,
                 max_episodes_per_type: Optional[int] = None,
                 goal_future: float = 0.98,
                 p_randomgoal: float = 0.5,
                 obs_horizon: int = 5,
                 load_actions: bool = False,
                 load_rewards: bool = False):
        self.data_dir = data_dir
        self.max_episodes_per_type = max_episodes_per_type
        self.goal_future = float(goal_future)
        self.p_randomgoal = float(p_randomgoal)
        self.req_horizon = int(max(1, min(5, obs_horizon)))
        self._load_actions = bool(load_actions)
        self._load_rewards = bool(load_rewards)

        const_zarr = os.path.join(self.data_dir, "constant.zarr")
        switch_zarr = os.path.join(self.data_dir, "switch.zarr")
        self.is_zarr = os.path.isdir(const_zarr) or os.path.isdir(switch_zarr)
        if not self.is_zarr:
            raise FileNotFoundError(f"Zarr datasets not found in {self.data_dir}. Expected constant.zarr and/or switch.zarr")

        want_types = types if types is not None else [t for t, p in [("constant", const_zarr), ("switch", switch_zarr)] if os.path.isdir(p)]
        if len(want_types) == 0:
            raise FileNotFoundError(f"No zarr datasets found under {self.data_dir}")

        self._buffers: List[Tuple[str, dict]] = []
        for t in want_types:
            zpath = os.path.join(self.data_dir, f"{t}.zarr")
            if not os.path.isdir(zpath):
                continue
            buf = ReplayBuffer.create_from_path(zpath, mode="r")
            new_buffer_dict = {
                "proprio": buf["proprio"][:],
                "episode_ends": buf.episode_ends[:]
            }
            if self._load_actions and ("actions" in buf):
                new_buffer_dict["actions"] = buf["actions"][:]
            if self._load_rewards and ("rewards" in buf):
                new_buffer_dict["rewards"] = buf["rewards"][:]
            self._buffers.append((t, new_buffer_dict))

        if len(self._buffers) == 0:
            raise FileNotFoundError(f"No valid zarr buffers opened from {self.data_dir}")

        # infer dims
        first_buf = self._buffers[0][1]
        prop_arr = first_buf["proprio"]
        if prop_arr.ndim == 3:
            avail_horizon = int(prop_arr.shape[1])
            feat_dim = int(prop_arr.shape[2])
        elif prop_arr.ndim == 2:
            avail_horizon = 1
            feat_dim = int(prop_arr.shape[1])
        else:
            raise ValueError(f"Unsupported proprio ndim: {prop_arr.ndim}")
        self.eff_horizon = int(min(self.req_horizon, avail_horizon))
        self.obs_dim = int(self.eff_horizon * feat_dim)

        # build step index (buf_id, t, ep_end)
        self._step_index: List[Tuple[int, int, int]] = []
        for buf_id, (tname, buf) in enumerate(self._buffers):
            ep_ends = np.asarray(buf['episode_ends'])
            num_eps_total = len(ep_ends)
            max_eps = self.max_episodes_per_type if self.max_episodes_per_type is not None else num_eps_total
            max_eps = min(max_eps, num_eps_total)
            prev_end = 0
            for ep_idx in range(max_eps):
                ep_end = int(ep_ends[ep_idx])
                ep_start = int(prev_end)
                prev_end = ep_end
                ep_len = ep_end - ep_start
                if ep_len < 2:
                    continue
                for t in range(ep_start, ep_end - 1):
                    self._step_index.append((buf_id, t, ep_end - 1))

        self.total_samples = len(self._step_index)
        # infer action dim if loaded
        self.action_dim: Optional[int] = None
        if self._load_actions:
            for _, buf in self._buffers:
                if "actions" in buf:
                    actions_arr = np.asarray(buf["actions"])
                    if actions_arr.ndim == 2:
                        self.action_dim = int(actions_arr.shape[1])
                    else:
                        # actions should be (T, A)
                        self.action_dim = int(actions_arr.shape[-1])
                    break

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        buf_id, t, ep_end = self._step_index[idx]
        _, buf = self._buffers[buf_id]

        # obs and next_obs
        prop_t = np.asarray(buf["proprio"][t])
        prop_t1 = np.asarray(buf["proprio"][t + 1])
        if prop_t.ndim == 2:
            obs_t = prop_t[-self.eff_horizon:, :].reshape(-1)
            next_obs_t = prop_t1[-self.eff_horizon:, :].reshape(-1)
        else:
            obs_t = prop_t.reshape(-1)
            next_obs_t = prop_t1.reshape(-1)

        # future goal (same-trajectory)
        u = np.random.rand()
        k = int(np.ceil(np.log(1 - u) / np.log(float(self.goal_future))))
        assert k > 0, f"k {k} must be greater than 0"
        future_idx = min(t + k, ep_end)
        prop_fut = np.asarray(buf["proprio"][future_idx])
        if prop_fut.ndim == 2:
            future_obs_t = prop_fut[-self.eff_horizon:, :].reshape(-1)
        else:
            future_obs_t = prop_fut.reshape(-1)

        # global random goal or same-trajectory goal
        if np.random.rand() <= self.p_randomgoal:
            rbi, rt, _ = self._step_index[np.random.randint(0, len(self._step_index))]
            rprop = np.asarray(self._buffers[rbi][1]["proprio"][rt])
            if rprop.ndim == 2:
                goal_obs_t = rprop[-self.eff_horizon:, :].reshape(-1)
            else:
                goal_obs_t = rprop.reshape(-1)
        else:
            goal_obs_t = future_obs_t

        out = {
            'obs': torch.as_tensor(obs_t, dtype=torch.float32),
            'next_obs': torch.as_tensor(next_obs_t, dtype=torch.float32),
            'goal_obs': torch.as_tensor(goal_obs_t, dtype=torch.float32),
        }

        if self._load_actions and ("actions" in buf):
            # Align action with transition (obs_t -> next_obs_{t+1}); use action at index t
            act_t = np.asarray(buf["actions"][t])
            out['action'] = torch.as_tensor(act_t, dtype=torch.float32)

        if self._load_rewards and ("rewards" in buf):
            rew_t = float(np.asarray(buf["rewards"][t]))
            out['reward'] = torch.as_tensor([rew_t], dtype=torch.float32)

        return out



class LeggedGymReplayLoader:
    """
    Adapter that exposes HilbertRepresentationDataset as an EpisodeBatch sampler
    compatible with URL-benchmark agents.

    Produces batches with fields: obs, action, reward, discount, next_obs, future_obs.
    - action: zeros of shape [B, action_dim] (legged dataset has no actions here)
    - reward: zeros [B, 1] (not used by HILP training)
    - discount: ones [B, 1] scaled by provided discount
    - future_obs: dataset-provided goal observation
    """

    def __init__(
        self,
        *,
        data_dir: str,
        action_dim: int,
        discount: float = 0.98,
        future: float = 0.99,
        p_currgoal: float = 0.0,
        p_randomgoal: float = 0.5,
        obs_horizon: int = 5,
        types: Optional[List[str]] = None,
        max_episodes_per_type: Optional[int] = None,
        use_actions: bool = False,
        use_rewards: bool = False,
    ) -> None:
        self._dataset = HilbertRepresentationDataset(
            data_dir=data_dir,
            types=types,
            max_episodes_per_type=max_episodes_per_type,
            goal_future=future,
            p_randomgoal=p_randomgoal,
            obs_horizon=obs_horizon,
            load_actions=use_actions,
            load_rewards=use_rewards,
        )
        self._discount = float(discount)
        self._future = float(future)
        self._p_currgoal = float(p_currgoal)
        self._p_randomgoal = float(p_randomgoal)
        self._frame_stack = None
        self._action_dim = int(action_dim)
        self._use_actions = bool(use_actions)
        self._use_rewards = bool(use_rewards)

    def __len__(self) -> int:
        return len(self._dataset)

    def sample(self, batch_size: int, custom_reward: Optional[object] = None, with_physics: bool = False) -> EpisodeBatch:
        del custom_reward
        del with_physics
        idxs = np.random.randint(0, len(self._dataset), size=int(batch_size))
        obs_list = []
        next_obs_list = []
        future_obs_list = []
        for idx in idxs:
            item = self._dataset[int(idx)]
            obs_list.append(item['obs'])
            next_obs_list.append(item['next_obs'])
            future_obs_list.append(item['goal_obs'])

        obs = torch.stack(obs_list, dim=0)
        next_obs = torch.stack(next_obs_list, dim=0)
        future_obs = torch.stack(future_obs_list, dim=0)

        if self._use_actions and ('action' in item):
            action = torch.stack([self._dataset[i]['action'] for i in idxs], dim=0)
        else:
            action = torch.zeros((obs.shape[0], self._action_dim), dtype=torch.float32)

        if self._use_rewards and ('reward' in item):
            reward = torch.stack([self._dataset[i]['reward'] for i in idxs], dim=0)
        else:
            reward = torch.zeros((obs.shape[0], 1), dtype=torch.float32)

        discount = torch.full((obs.shape[0], 1), fill_value=self._discount, dtype=torch.float32)

        return EpisodeBatch(
            obs=obs,
            action=action,
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            future_obs=future_obs,
            meta={},
        )

