#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import zarr

from replay_buffer import ReplayBuffer


class HilbertRepresentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 types: Optional[List[str]] = None,
                 max_episodes_per_type: Optional[int] = None,
                 goal_future: float = 0.98,
                 p_trajgoal: float = 0.5,
                 p_randomgoal: float = 0.5,
                 obs_horizon: int = 5):
        self.data_dir = data_dir
        self.max_episodes_per_type = max_episodes_per_type
        self.goal_future = float(goal_future)
        self.p_trajgoal = float(p_trajgoal)
        self.p_randomgoal = float(p_randomgoal)
        assert self.p_trajgoal + self.p_randomgoal <= 1.0, "p_trajgoal + p_randomgoal must be <= 1.0"
        print("Sum of p_trajgoal and p_randomgoal is ", self.p_trajgoal + self.p_randomgoal)
        self.req_horizon = int(max(1, min(5, obs_horizon)))

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
        future_idx = min(t + 1 + k, ep_end)
        prop_fut = np.asarray(buf["proprio"][future_idx])
        if prop_fut.ndim == 2:
            future_obs_t = prop_fut[-self.eff_horizon:, :].reshape(-1)
        else:
            future_obs_t = prop_fut.reshape(-1)

        # global random goal or same-trajectory goal
        if np.random.rand() >= self.p_trajgoal:
            rbi, rt, _ = self._step_index[np.random.randint(0, len(self._step_index))]
            rprop = np.asarray(self._buffers[rbi][1]["proprio"][rt])
            if rprop.ndim == 2:
                goal_obs_t = rprop[-self.eff_horizon:, :].reshape(-1)
            else:
                goal_obs_t = rprop.reshape(-1)
        else:
            goal_obs_t = future_obs_t

        return {
            'obs': torch.as_tensor(obs_t, dtype=torch.float32),
            'next_obs': torch.as_tensor(next_obs_t, dtype=torch.float32),
            'goal_obs': torch.as_tensor(goal_obs_t, dtype=torch.float32),
        }


