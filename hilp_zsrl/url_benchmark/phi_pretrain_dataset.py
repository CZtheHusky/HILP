import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm, trange
import os
import json
from typing import List, Tuple, Optional, Dict
import h5py


class PhiDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir: str, 
        obs_horizon: int = 1,
        total_episodes: int = 10000, 
        discount: float = 0.98, 
        goal_future: float = 0.98,
        p_randomgoal: float = 0.375,
    ):
        self.data_dir = data_dir
        self.meta_info = {
            "total_episodes": total_episodes,
            "p_randomgoal": p_randomgoal,
            "goal_future": goal_future,
            "discount": discount,
            "obs_horizon": obs_horizon,
            "loading_keys": ['proprio', 'actions'],
            "current_write_p": 0,
            "valid_episodes_num": 0,
        }
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.data_dir, "meta.json")):
            self._save_meta_info()
        else:
            self._load_meta_info()
        self.tmp_data = defaultdict(list)
        self.tmp_meta = defaultdict(list)
        self.tmp_episode_len = []

        # self._load_dataset()
        # self._build_step_index()

    def _load_meta_info(self):
        with open(os.path.join(self.data_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        for k, v in self.meta_info.items():
            assert k in meta, f"Key {k} not found in meta.json"
            if v != meta[k]:
                print(f"Key {k} in meta.json is different from the initial value, meta: {meta[k]} -VS- init: {v}\nOverwriting with the value from meta.json")
                self.meta_info[k] = meta[k]

    def _save_meta_info(self):
        with open(os.path.join(self.data_dir, "meta.json"), "w") as f:
            json.dump(self.meta_info, f)

    def _collect_and_save(self, episode: List[Dict[str, np.ndarray]]):
        for ep_data in episode:
            data = {k:v for k, v in ep_data['data'].items() if k in self.meta_info['loading_keys']}
            ep_len = data['proprio'].shape[0]
            current_write_p = self.meta_info['current_write_p']
            meta = ep_data['meta']
            if self.meta_info['valid_episodes_num'] < self.meta_info['total_episodes']:
                self.tmp_episode_len.append(ep_len)
                for k in data.keys():
                    self.tmp_data[k].append(data[k])
                for k in meta.keys():
                    self.tmp_meta[k].append(meta[k])
            else:
                self.tmp_episode_len[current_write_p] = ep_len
                for k in ep_data.keys():
                    self.tmp_data[k][current_write_p] = data[k]
                for k in meta.keys():
                    self.tmp_meta[k][current_write_p] = meta[k]
            self.meta_info['current_write_p'] += 1
            self.meta_info['valid_episodes_num'] = max(self.meta_info['valid_episodes_num'], min(self.meta_info['current_write_p'], self.meta_info['total_episodes']))
            self.meta_info['current_write_p'] %= self.meta_info['total_episodes']
            

    def _in_memory_finalize(self):
        for k, v in self.data.items():
            self.tmp_data[k] = np.concatenate(v, axis=0)
        for k, v in self.meta.items():
            self.tmp_meta[k] = np.stack(v, axis=0)
        episode_ends = np.cumsum(self.tmp_episode_len)
        self.data = self.tmp_data
        self.meta = self.tmp_meta
        self.data['episode_ends'] = episode_ends
        self.data['episode_len'] = np.array(self.tmp_episode_len)
        episode_id = np.repeat(np.arange(len(episode_ends)), np.diff([0, *episode_ends]))
        self.data['episode_id'] = episode_id
        self.tmp_data = defaultdict(list)
        self.tmp_meta = defaultdict(list)
        self.tmp_episode_len = []
        self._save_meta_info()
        self._build_step_index()

    def _in_memory_split(self):
        if hasattr(self, 'data'):
            self.tmp_data = defaultdict(list)
            self.tmp_meta = defaultdict(list)
            self.tmp_episode_len = self.data['episode_len'].tolist()
            for ep_id in range(len(self.data['episode_ends'])):
                data_start_idx = 0 if ep_id == 0 else self.data['episode_ends'][ep_id - 1]
                data_end_idx = self.data['episode_ends'][ep_id]
                for k in self.meta_info['loading_keys']:
                    self.tmp_data[k].append(self.data[k][data_start_idx:data_end_idx])
                for k in self.meta.keys():
                    self.tmp_meta[k].append(self.meta[k][ep_id])
            del self.data
            del self.meta
        else:
            self.tmp_data = defaultdict(list)
            self.tmp_meta = defaultdict(list)
            self.tmp_episode_len = []


    def _load_dataset(self):
        raise NotImplementedError()
        self.data = defaultdict(list)
        self.meta = defaultdict(list)
        npz_files = os.listdir(self.data_dir)
        if len(npz_files) == 0:
            return
        npz_files = [f for f in npz_files if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(x.split(".")[0]))
        pbar = trange(len(npz_files), desc="Loading dataset", leave=True)
        episode_len = []
        for npz_file in tqdm(npz_files[:self.meta_info['total_episodes']]):
            with open(os.path.join(self.data_dir, npz_file), "rb") as f:
                loaded = np.load(f)
                ep_data = {k: loaded[k] for k in self.meta_info['loading_keys']}
                ep_meta = {k.replace("meta/", ""): loaded[k] for k in loaded.files if k.startswith("meta/")}
            ep_len = ep_data['proprio'].shape[0]
            episode_len.append(ep_len)
            pbar.update(1)
            pbar.set_description(f"Loading: {npz_file}")
            for k in ep_data.keys():
                self.data[k].append(ep_data[k])
            for k in ep_meta.keys():
                self.meta[k].append(ep_meta[k])
        pbar.close()
        for k, v in self.data.items():
            self.data[k] = np.concatenate(v, axis=0)
        for k, v in self.meta.items():
            self.meta[k] = np.stack(v, axis=0)
        episode_ends = np.cumsum(episode_len)
        self.data['episode_ends'] = episode_ends
        self.data['episode_len'] = np.array(episode_len)
        episode_id = np.repeat(np.arange(len(episode_ends)), np.diff([0, *episode_ends]))
        self.data['episode_id'] = episode_id


    def _build_step_index(self):
        self._step_index: List[Tuple[int, int]] = []
        ep_ends = self.data['episode_ends']
        if "ep_start_obs" not in self.meta:
            ep_start_delta = self.meta_info['obs_horizon'] - 1
        else:
            ep_start_delta = max(0, self.meta_info['obs_horizon'] - 5)
        num_eps_total = len(ep_ends)
        for ep_idx in range(int(num_eps_total)):
            ep_end = int(ep_ends[ep_idx])
            ep_start = 0 if ep_idx == 0 else ep_ends[ep_idx - 1]
            ep_len = ep_end - ep_start
            if ep_len < self.meta_info['obs_horizon'] + 1:
                continue
            for t in range(ep_start + ep_start_delta, ep_end - 1):
                self._step_index.append((t, ep_end - 1))

    def __len__(self):
        return len(self._step_index)

    def get_obs(self, t):
        ep_id = self.data['episode_id'][t]
        ep_start = 0 if ep_id == 0 else self.data['episode_ends'][ep_id - 1]
        horizon_start = max(ep_start, t - self.meta_info['obs_horizon'] + 1)
        horizon_end = t + 1
        proprio = self.data['proprio'][horizon_start:horizon_end]
        valid_len = proprio.shape[0]
        obs = np.empty((self.meta_info['obs_horizon'], proprio.shape[-1]), dtype=proprio.dtype)
        pad = self.meta_info['obs_horizon'] - valid_len
        obs[pad:] = proprio
        if pad > 0:
            obs[:pad] = self.meta["ep_start_obs"][ep_id, -pad:, :obs.shape[-1]]
        return obs.reshape(-1)

    def __getitem__(self, idx):
        t, ep_end = self._step_index[idx]
        obs_t = self.get_obs(t)
        next_obs_t = self.get_obs(t + 1)
        if np.random.rand() <= self.meta_info['p_randomgoal']:
            rt, _ = self._step_index[np.random.randint(0, self.__len__())]
            rprop = self.get_obs(rt)
            goal_obs_t = rprop
        else:
            k = np.random.geometric(p=1-self.meta_info['goal_future'])
            assert k > 0, f"k {k} must be greater than 0"
            future_idx = min(t + k, ep_end)
            goal_obs_t = self.get_obs(future_idx)

        out = {
            'obs': obs_t.astype(np.float32),
            'next_obs': next_obs_t.astype(np.float32),
            'future_obs': goal_obs_t.astype(np.float32),
        }
        if self.meta_info['discount'] is not None:
            out['discount'] = self.meta_info['discount']
        return out

