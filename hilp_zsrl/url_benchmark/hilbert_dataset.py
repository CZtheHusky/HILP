#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import zarr
from tqdm import tqdm
from url_benchmark.replay_buffer import DataBuffer
from url_benchmark.in_memory_replay_buffer import EpisodeBatch



class HugWBCSLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        types: Optional[List[str]] = None,
        max_episodes_per_type: Optional[int] = None,
        obs_horizon: int = 5,
    ):
        self.data_dir = data_dir
        self.max_episodes_per_type = max_episodes_per_type
        self.req_horizon = int(max(1, min(5, obs_horizon)))
        const_zarr = os.path.join(self.data_dir, "constant.zarr")
        switch_zarr = os.path.join(self.data_dir, "switch.zarr")
        self.is_zarr = os.path.isdir(const_zarr) or os.path.isdir(switch_zarr)
        if "large" in self.data_dir:
            full_loading = False
        else:
            full_loading = True
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
            buf = DataBuffer.create_from_path(zpath, mode="r")
            new_buffer_dict = {}
            if full_loading:
                new_buffer_dict['proprio'] = buf["proprio"][:]
                new_buffer_dict['actions'] = buf['actions'][:]
                new_buffer_dict['rewards'] = buf['rewards'][:]
                new_buffer_dict['privileged_obs'] = buf['critic_obs'][:]
                new_buffer_dict["episode_ends"] = buf.episode_ends[:]
                new_buffer_dict['z_vector'] = buf['commands'][:, -1, :]
                new_buffer_dict['clock'] = buf['clock'][:, -1, :]
            else:
                new_buffer_dict['proprio'] = buf["proprio"]
                new_buffer_dict['actions'] = buf['actions'][:]
                new_buffer_dict['rewards'] = buf['rewards'][:]
                new_buffer_dict['privileged_obs'] = buf['critic_obs']
                new_buffer_dict["episode_ends"] = buf.episode_ends[:]
                new_buffer_dict['z_vector'] = buf['commands'][:, -1, :]
                new_buffer_dict['clock'] = buf['clock'][:, -1, :]
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
                for t in range(ep_start, ep_end):
                    self._step_index.append((buf_id, t))

        self.total_samples = len(self._step_index)
        # infer action dim if loaded
        self.action_dim: Optional[int] = 19
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        buf_id, t = self._step_index[idx]
        _, buf = self._buffers[buf_id]
                
        out = {
            'obs': torch.as_tensor(buf["proprio"][t], dtype=torch.float32),
            'actions': torch.as_tensor(buf['actions'][t], dtype=torch.float32),
            'rewards': torch.as_tensor(buf['rewards'][t], dtype=torch.float32),
            'privileged_obs': torch.as_tensor(buf['privileged_obs'][t], dtype=torch.float32),
            'z_vector': torch.as_tensor(buf['z_vector'][t], dtype=torch.float32),
            'clock': torch.as_tensor(buf['clock'][t], dtype=torch.float32),
        }
        return out


class HilbertRepresentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 goal_future: float = 0.98,
                 p_randomgoal: float = 0.5,
                 obs_horizon: int = 5,
                 full_loading: bool = False,
                 use_history_action: bool = True,
                 discount: float = None,
                 load_command: bool = False,
        ):
        print(f"Sampling method, p_randomgoal: {p_randomgoal}, future: {goal_future}, discount: {discount} horizon: {obs_horizon} use_history_action: {use_history_action}")
        self.data_dir = data_dir
        self.goal_future = float(goal_future)
        self.p_randomgoal = float(p_randomgoal)
        self.req_horizon = obs_horizon
        self._full_loading = bool(full_loading)
        self._discount = np.array(discount, dtype=np.float32)
        self.use_history_action = use_history_action
        self._load_command = load_command

        self._buffers: List[Tuple[str, dict]] = []
        for dirn in tqdm(os.listdir(self.data_dir)):
            sub_path = os.path.join(self.data_dir, dirn)
            if not os.path.isdir(sub_path):
                continue
            if sub_path.endswith("zarr"):
                self._load_data_from_rb(sub_path, use_history_action)
            else:
                for zarr_dir in os.listdir(sub_path):
                    if not zarr_dir.endswith("zarr"):
                        continue
                    self._load_data_from_rb(os.path.join(sub_path, zarr_dir), use_history_action)

        # infer dims
        first_buf = self._buffers[0][1]
        feat_dim = int(first_buf["proprio"].shape[-1] + first_buf["actions"].shape[-1] if use_history_action else first_buf["proprio"].shape[-1])
        self.obs_dim = int(self.req_horizon * feat_dim)

        # build step index (buf_id, t, ep_end)
        self._step_index: List[Tuple[int, int, int]] = []
        for buf_id, (tname, buf) in enumerate(self._buffers):
            ep_ends = np.asarray(buf['episode_ends'])
            if "ep_start_obs" not in buf:
                ep_start_delta = 4
            else:
                ep_start_delta = 0
            num_eps_total = len(ep_ends)
            for ep_idx in range(num_eps_total):
                ep_end = int(ep_ends[ep_idx])
                ep_start = 0 if ep_idx == 0 else ep_ends[ep_idx - 1] + ep_start_delta
                ep_len = ep_end - ep_start
                if ep_len < self.req_horizon:
                    continue
                for t in range(ep_start + ep_start_delta, ep_end - 1):
                    self._step_index.append((buf_id, t, ep_end - 1))

        self.total_samples = len(self._step_index)
        self._step_index = np.asarray(self._step_index, dtype=np.int32)
        # infer action dim if loaded
        self.action_dim: Optional[int] = None
        if self._full_loading:
            for _, buf in self._buffers:
                if "actions" in buf:
                    actions_arr = np.asarray(buf["actions"])
                    self.action_dim = int(actions_arr.shape[-1])

    def _load_data_from_rb(self, zpath, use_history_action):
        buf = DataBuffer.create_from_path(zpath, mode="r")
        print(buf)
        new_buffer_dict = {
            "proprio": buf["proprio"][:],
        }
        if use_history_action:
            new_buffer_dict['actions'] = buf['actions'][:]
        obs_dim = int(buf["proprio"].shape[-1] + buf["actions"].shape[-1] if use_history_action else buf["proprio"].shape[-1])
        new_buffer_dict['ep_start_obs'] = buf.meta['ep_start_obs'][..., :obs_dim]
        if self._full_loading:
            if not use_history_action:
                new_buffer_dict['actions'] = buf['actions'][:]
            # new_buffer_dict['rewards'] = buf['rewards'][:]
            new_buffer_dict['privileged_obs'] = buf['privileged'][:, :3]
            if self._load_command:
                new_buffer_dict['commands'] = buf['commands'][:]
        new_buffer_dict["episode_ends"] = buf.episode_ends[:]
        episode_id = np.repeat(np.arange(len(new_buffer_dict["episode_ends"])), np.diff([0, *new_buffer_dict["episode_ends"]]))
        new_buffer_dict['episode_id'] = episode_id
        self._buffers.append((zpath, new_buffer_dict))

    def __len__(self):
        return self.total_samples


    def get_obs(self, buf, t):
        ep_id = buf['episode_id'][t]
        ep_start = 0 if ep_id == 0 else buf['episode_ends'][ep_id - 1]
        horizon_start = max(ep_start, t - self.req_horizon + 1)
        horizon_end = t + 1
        proprio = buf['proprio'][horizon_start:horizon_end]
        valid_len = proprio.shape[0]
        # if self.use_history_action:
        #     history_action = np.zeros((valid_len, buf['actions'].shape[-1]))
        #     if valid_len > 1:
        #         his_a_start = max(ep_start, t - self.req_horizon)
        #         his_a_end = t
        #         history_len = his_a_end - his_a_start
        #         history_action[-history_len:] = buf['actions'][his_a_start:his_a_end]
        #     obs = np.concatenate([proprio, history_action], axis=-1)
        # else:
        #     obs = proprio
        # if valid_len < self.req_horizon:
        #     obs = np.concatenate([buf['ep_start_obs'][ep_id, -(self.req_horizon - valid_len):, :obs.shape[-1]], obs], axis=0)
        # return obs.reshape(-1)
        if self.use_history_action:
            # 预分配 + 原地写，避免 concatenate
            act_dim = buf["actions"].shape[-1]
            obs = np.empty((self.req_horizon, proprio.shape[-1] + act_dim), dtype=proprio.dtype)
            # 前段可能需要 pad ep_start_obs
            pad = self.req_horizon - valid_len
            obs[pad:pad+valid_len, :proprio.shape[-1]] = proprio
            # history actions
            obs[:, proprio.shape[-1]:] = 0
            if valid_len > 1:
                his_a_start = max(ep_start, t - self.req_horizon)
                his_a_end = t
                history = buf["actions"][his_a_start:his_a_end]
                obs[-history.shape[0]:, proprio.shape[-1]:] = history
            if pad > 0:
                obs[:pad] = buf["ep_start_obs"][ep_id, -pad:, :obs.shape[-1]]
        else:
            obs = np.empty((self.req_horizon, proprio.shape[-1]), dtype=proprio.dtype)
            pad = self.req_horizon - valid_len
            obs[pad:pad+valid_len] = proprio
            if pad > 0:
                obs[:pad] = buf["ep_start_obs"][ep_id, -pad:, :obs.shape[-1]]

        return obs.reshape(-1)

    def __getitem__(self, idx):
        buf_id, t, ep_end = self._step_index[idx]
        _, buf = self._buffers[buf_id]

        obs_t = self.get_obs(buf, t)
        next_obs_t = self.get_obs(buf, t + 1)

        k = np.random.geometric(p=1-self.goal_future)
        assert k > 0, f"k {k} must be greater than 0"
        future_idx = min(t + k, ep_end)
        future_obs_t = self.get_obs(buf, future_idx)

        if np.random.rand() <= self.p_randomgoal:
            rbi, rt, _ = self._step_index[np.random.randint(0, self.total_samples)]
            rprop = self.get_obs(self._buffers[rbi][1], rt)
            goal_obs_t = rprop
        else:
            goal_obs_t = future_obs_t

        out = {
            'obs': obs_t.astype(np.float32),
            'next_obs': next_obs_t.astype(np.float32),
            'future_obs': goal_obs_t.astype(np.float32),
        }

        if self._full_loading:
            out['actions'] = buf['actions'][t].astype(np.float32)
            # out['rewards'] = buf['rewards'][t]
            out['privileged_obs'] = buf['privileged_obs'][t].astype(np.float32)
            if "commands" in buf:
                out['commands'] = buf['commands'][t].astype(np.float32)
        if self._discount is not None:
            out['discount'] = self._discount
        return out
   

class HilbertRepresentationDatasetLegacy(Dataset):
    def __init__(self, data_dir: str,
                 types: Optional[List[str]] = None,
                 max_episodes_per_type: Optional[int] = None,
                 goal_future: float = 0.98,
                 p_randomgoal: float = 0.5,
                 obs_horizon: int = 5,
                 full_loading: bool = False,
                 use_history_action: bool = True,
                 discount: float = None,
        ):
        print(f"Sampling method, p_randomgoal: {p_randomgoal}, future: {goal_future}, discount: {discount} horizon: {obs_horizon} use_history_action: {use_history_action}")
        self.data_dir = data_dir
        self.max_episodes_per_type = max_episodes_per_type
        self.goal_future = float(goal_future)
        self.p_randomgoal = float(p_randomgoal)
        self.req_horizon = int(max(1, min(5, obs_horizon)))
        self._full_loading = bool(full_loading)
        self._discount = discount

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
            buf = DataBuffer.create_from_path(zpath, mode="r")
            print(buf)
            new_buffer_dict = {
                "proprio": buf["proprio"][:],
            }
            if self._full_loading:
                new_buffer_dict['actions'] = buf['actions'][:]
                new_buffer_dict['rewards'] = buf['rewards'][:]
                new_buffer_dict['privileged_obs'] = buf['privileged'][:, :3]
                # new_buffer_dict['commands'] = buf['commands'][:]
            new_buffer_dict["episode_ends"] = buf.episode_ends[:]
            if "ep_start_obs" in buf.meta:
                new_buffer_dict['ep_start_obs'] = buf['ep_start_obs'][:]
            self._buffers.append((t, new_buffer_dict))

        if len(self._buffers) == 0:
            raise FileNotFoundError(f"No valid zarr buffers opened from {self.data_dir}")

        # infer dims
        first_buf = self._buffers[0][1]
        feat_dim = int(first_buf["proprio"].shape[-1] + first_buf["actions"].shape[-1] if use_history_action else first_buf["proprio"].shape[-1])
        self.obs_dim = int(self.req_horizon * feat_dim)

        # build step index (buf_id, t, ep_end)
        self._step_index: List[Tuple[int, int, int]] = []
        for buf_id, (tname, buf) in enumerate(self._buffers):
            ep_ends = np.asarray(buf['episode_ends'])
            if "ep_start_obs" not in buf:
                ep_start_delta = 4
            num_eps_total = len(ep_ends)
            max_eps = self.max_episodes_per_type if self.max_episodes_per_type is not None else num_eps_total
            max_eps = min(max_eps, num_eps_total)
            for ep_idx in range(max_eps):
                ep_end = int(ep_ends[ep_idx])
                ep_start = 0 if ep_idx == 0 else ep_ends[ep_idx - 1]
                ep_len = ep_end - ep_start
                if ep_len < self.req_horizon:
                    continue
                for t in range(ep_start + ep_start_delta, ep_end - 1):
                    self._step_index.append((buf_id, t, ep_end - 1))

        self.total_samples = len(self._step_index)
        # infer action dim if loaded
        self.action_dim: Optional[int] = None
        if self._full_loading:
            for _, buf in self._buffers:
                if "actions" in buf:
                    actions_arr = np.asarray(buf["actions"])
                    self.action_dim = int(actions_arr.shape[-1])
                    break

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        buf_id, t, ep_end = self._step_index[idx]
        _, buf = self._buffers[buf_id]

        # obs and next_obs
        prop_t = np.asarray(buf["proprio"][t-self.req_horizon+1:t+1])
        prop_t1 = np.asarray(buf["proprio"][t-self.req_horizon+2:t+2])


        obs_t = prop_t.reshape(-1)
        next_obs_t = prop_t1.reshape(-1)

        # # future goal (same-trajectory)
        # u = np.random.rand()
        # k = int(np.ceil(np.log(1 - u) / np.log(float(self.goal_future))))
        # assert k > 0, f"k {k} must be greater than 0"
        k = np.random.geometric(p=1-self.goal_future)
        assert k > 0, f"k {k} must be greater than 0"
        future_idx = min(t + k, ep_end)
        prop_fut = np.asarray(buf["proprio"][future_idx-self.req_horizon+1:future_idx+1])
        future_obs_t = prop_fut.reshape(-1)

        # global random goal or same-trajectory goal
        if np.random.rand() <= self.p_randomgoal:
            rbi, rt, _ = self._step_index[np.random.randint(0, self.total_samples)]
            rprop = np.asarray(self._buffers[rbi][1]["proprio"][rt-self.req_horizon+1:rt+1])
            goal_obs_t = rprop.reshape(-1)
        else:
            goal_obs_t = future_obs_t

        out = {
            'obs': obs_t,
            'next_obs': next_obs_t,
            'future_obs': goal_obs_t,
        }

        if self._full_loading:
            out['actions'] = buf['actions'][t]
            out['rewards'] = buf['rewards'][t]
            out['privileged_obs'] = buf['privileged_obs'][t]
            if "commands" in buf:
                out['commands'] = buf['commands'][t]
        if self._discount is not None:
            out['discount'] = self._discount
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
        full_loading: bool = False,
        split_val_set: bool = False,
        use_history_action: bool = True,
    ) -> None:
        print(f"Sampling method, p_randomgoal: {p_randomgoal}, p_currgoal: {p_currgoal}, future: {future}, discount: {discount}")
        self._dataset = HilbertRepresentationDataset(
            data_dir=data_dir,
            types=types,
            max_episodes_per_type=max_episodes_per_type,
            goal_future=future,
            p_randomgoal=p_randomgoal,
            obs_horizon=obs_horizon,
            full_loading=full_loading,
            use_history_action=use_history_action,
        )
        self._discount = float(discount)
        self._future = float(future)
        self._p_currgoal = float(p_currgoal)
        self._p_randomgoal = float(p_randomgoal)
        self._frame_stack = None
        self._action_dim = int(action_dim)
        self._full_loading = bool(full_loading)
        if split_val_set:
            self.train_index = np.random.randint(0, len(self._dataset), size=int(len(self._dataset) * 0.9))
            self.val_index = np.setdiff1d(np.arange(len(self._dataset)), self.train_index)
        self._split_val_set = split_val_set

    def __len__(self) -> int:
        if self._split_val_set:
            return len(self.train_index)
        else:
            return len(self._dataset)

    def sample(self, batch_size: int, custom_reward: Optional[object] = None, with_physics: bool = False, is_val=False) -> EpisodeBatch:
        del custom_reward
        del with_physics
        if is_val and self._split_val_set:
            all_idxs = self.val_index
        elif not is_val and self._split_val_set:
            all_idxs = self.train_index
        else:
            all_idxs = np.arange(len(self._dataset))
        
        idxs = np.random.randint(0, len(all_idxs), size=int(batch_size))
        obs_list = []
        next_obs_list = []
        future_obs_list = []
        for idx in idxs:
            item = self._dataset[all_idxs[int(idx)]]
            obs_list.append(item['obs'])
            next_obs_list.append(item['next_obs'])
            future_obs_list.append(item['future_obs'])

        obs = torch.stack(obs_list, dim=0)
        next_obs = torch.stack(next_obs_list, dim=0)
        future_obs = torch.stack(future_obs_list, dim=0)

        if self._full_loading:
            action = torch.stack([self._dataset[i]['actions'] for i in idxs], dim=0)
            reward = torch.stack([self._dataset[i]['rewards'] for i in idxs], dim=0)
            privileged_obs = torch.stack([self._dataset[i]['privileged_obs'] for i in idxs], dim=0)
            commands = torch.stack([self._dataset[i]['commands'] for i in idxs], dim=0)
        else:
            action = torch.zeros((obs.shape[0], self._action_dim), dtype=torch.float32)
            reward = torch.zeros((obs.shape[0], 1), dtype=torch.float32)
            privileged_obs = torch.zeros((obs.shape[0], 3), dtype=torch.float32)
            commands = torch.zeros((obs.shape[0], 11), dtype=torch.float32)
        discount = torch.full((obs.shape[0], 1), fill_value=self._discount, dtype=torch.float32)

        return EpisodeBatch(
            obs=obs,
            action=action,
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            future_obs=future_obs,
            privileged_obs=privileged_obs,
            commands=commands,
            meta={},
        )

