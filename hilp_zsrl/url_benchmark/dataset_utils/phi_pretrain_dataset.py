import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm, trange
import os
import json
import typing as tp

from typing import List, Tuple, Optional, Dict
import h5py
from url_benchmark.dataset_utils.in_memory_replay_buffer import EpisodeBatch


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
            "current_size": 0,
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

    def add_episodes(self, episode: List[Dict[str, np.ndarray]]):
        for ep_data in episode:
            data = {k:v for k, v in ep_data['data'].items() if k in self.meta_info['loading_keys']}
            ep_len = data['proprio'].shape[0]
            current_write_p = self.meta_info['current_write_p']
            meta = ep_data['meta']
            if self.meta_info['current_size'] < self.meta_info['total_episodes']:
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
            self.meta_info['current_size'] = max(self.meta_info['current_size'], min(self.meta_info['current_write_p'], self.meta_info['total_episodes']))
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



class PhiWalkerDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir: str, 
        obs_horizon: int = 1,
        total_episodes: int = 10000, 
        discount: float = 0.98, 
        goal_future: float = 0.98,
        p_randomgoal: float = 0.375,
        random_sample: bool = False,
    ):
        self.data_dir = data_dir
        self.meta_info = {
            "total_episodes": total_episodes,
            "p_randomgoal": p_randomgoal,
            "goal_future": goal_future,
            "discount": discount,
            "obs_horizon": obs_horizon,
            "current_write_p": 0,
            "current_size": 0,
        }
        self.random_sample = random_sample
        os.makedirs(self.data_dir, exist_ok=True)
        # if not os.path.exists(os.path.join(self.data_dir, "meta.json")):
        #     self._save_meta_info()
        # else:
        #     self._load_meta_info()
        self.data_storage = {}
        self.meta_storage = {}

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

    def add_episodes(self, episode: Dict[str, np.ndarray], num_episodes: int):
        current_idx = np.arange(num_episodes) + self.meta_info['current_write_p']
        self.meta_info['current_write_p'] = self.meta_info['current_write_p'] + num_episodes
        self.meta_info['current_size'] = max(self.meta_info['current_size'], min(self.meta_info['current_write_p'], self.meta_info['total_episodes']))        
        self.meta_info['current_write_p'] %= self.meta_info['total_episodes']  
        current_idx %= self.meta_info['total_episodes']
        for k, v in episode.items():
            if k not in self.data_storage:
                self.data_storage[k] = np.empty((self.meta_info['total_episodes'], *v.shape[1:]), dtype=v.dtype)
                self.max_episode_steps = v.shape[1]
            self.data_storage[k][current_idx] = v
        self.dataset_len = self.meta_info['current_size'] * (self.max_episode_steps - 1)
        self.num_steps = self.meta_info['current_size'] * (self.max_episode_steps)

    def __len__(self):
        return self.dataset_len
    
    @property
    def ep_lengths(self):
        return self.max_episode_steps
    
    @property
    def valid_len(self):
        return self.max_episode_steps - 1

    @property
    def num_episodes(self):
        return self.meta_info['current_size']

    @property
    def obs_horizon(self):
        return self.meta_info['obs_horizon']

    def get_obs(self, ep_idx, t):
        if self.obs_horizon == 1:
            return self.data_storage['observation'][ep_idx, t]
        else:
            obs = np.empty((self.obs_horizon, self.data_storage['observation'].shape[-1]), dtype=self.data_storage['observation'].dtype)
            valid_obs = self.data_storage['observation'][ep_idx, max(0, t - self.obs_horizon + 1):t + 1]
            obs[self.obs_horizon - valid_obs.shape[0]:] = valid_obs
            obs[:self.obs_horizon - valid_obs.shape[0]] = valid_obs[0]
        return obs.reshape(-1)

    def get_episode(self, ep_idx=None, custom_reward=None):
        if ep_idx is None:
            ep_idx = np.random.randint(0, self.num_episodes)
        episode_traj = self.data_storage['observation'][ep_idx]
        if custom_reward is not None:
            phy = self.data_storage["physics"][ep_idx]
            ds_rewards = custom_reward.from_physics(phy)
            ds_rewards = np.array(ds_rewards, dtype=np.float32)
        else:
            ds_rewards = self.data_storage['reward'][ep_idx]
        return ep_idx, episode_traj, ds_rewards

    def sample(self, batch_size, custom_reward: tp.Optional[tp.Any] = None, with_physics: bool = False) -> EpisodeBatch:
        ep_idx = np.random.randint(0, self.num_episodes, size=batch_size)
        step_idx = np.random.randint(0, self.valid_len)
        obs = self.get_obs(ep_idx, step_idx)
        action = self.data_storage['action'][ep_idx, step_idx]
        next_obs = self.get_obs(ep_idx, step_idx + 1)
        phy = self.data_storage['physics'][ep_idx, step_idx]
        if custom_reward is not None:
            reward = np.array([[custom_reward.from_physics(p)] for p in phy], dtype=np.float32)
        else:
            reward = self.data_storage['reward'][ep_idx, step_idx]
        discount = self.meta_info['discount'] * np.ones_like(reward, dtype=np.float32)
        future_obs: tp.Optional[np.ndarray] = None
        if self.meta_info['goal_future'] < 1:
            future_idx = step_idx + np.random.geometric(p=(1 - self.meta_info['goal_future']), size=batch_size)
            future_idx = np.clip(future_idx, 0, self.valid_len)
            assert (future_idx <= self.valid_len).all()
            future_obs = self.get_obs(ep_idx, future_idx)
            mask = np.random.rand(batch_size) <= self.meta_info['p_randomgoal']
            mask_num = mask.sum()
            if mask_num > 0:
                random_ep_idx = np.random.randint(0, self.num_episodes, size=mask_num)
                random_step_idx = np.random.randint(0, self.valid_len, size=mask_num)
                random_obs = self.get_obs(random_ep_idx, random_step_idx)
                future_obs[mask] = random_obs
        additional = {}
        if with_physics:
            additional["_physics"] = phy
        return EpisodeBatch(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs, future_obs=future_obs, **additional)

    def sample_transitions_with_indices(self, batch_size: int, custom_reward: tp.Optional[tp.Any] = None) -> tp.Dict[str, np.ndarray]:
        """Sample a batch of transitions and also return (ep_idx, step_idx) for each sampled transition.
        step_idx corresponds to the next_obs time index in the stored episode (i.e., the transition s_t->s_{t+1} uses step_idx=t+1).
        """
        batch_size = min(batch_size, self.dataset_len)
        # ep_idx = np.random.randint(0, self.num_episodes, size=batch_size)
        # step_idx = np.random.randint(0, self.valid_len, size=batch_size)  # 1..len
        global_step_idx = np.random.randint(0, self.dataset_len, size=batch_size)
        ep_idx = global_step_idx // self.valid_len
        step_idx = global_step_idx % self.valid_len
        obs = self.get_obs(ep_idx, step_idx)
        # Fetch data exactly like sample()
        next_obs = self.get_obs(ep_idx, step_idx + 1)
        assert np.all(step_idx + 1 <= self.valid_len)
        phy = self.data_storage['physics'][ep_idx, step_idx]
        if custom_reward is not None:
            reward = np.array([[custom_reward.from_physics(p)] for p in phy], dtype=np.float32)
        else:
            reward = self.data_storage['reward'][ep_idx, step_idx]
        return {
            'obs': obs,
            'next_obs': next_obs,
            'reward': reward,
            'ep_idx': ep_idx,
            'step_idx': step_idx,
        }

    def __getitem__(self, idx):
        if self.random_sample:
            # print("Random sampling, size: ", len(self))
            idx = np.random.randint(0, len(self))
        ep_idx = idx // self.valid_len
        assert ep_idx < self.num_episodes
        t = idx % self.valid_len

        obs_t = self.get_obs(ep_idx, t)
        next_obs_t = self.get_obs(ep_idx, t + 1)
        if np.random.rand() <= self.meta_info['p_randomgoal']:
            r_idx = np.random.randint(0, self.num_steps)
            r_ep_idx = r_idx // self.ep_lengths
            rt = r_idx % self.ep_lengths
            rprop = self.get_obs(r_ep_idx, rt)
            goal_obs_t = rprop
        else:
            k = np.random.geometric(p=1-self.meta_info['goal_future'])
            assert k > 0, f"k {k} must be greater than 0"
            future_idx = min(t + k, self.valid_len)
            goal_obs_t = self.get_obs(ep_idx, future_idx)

        out = {
            'obs': obs_t.astype(np.float32),
            'next_obs': next_obs_t.astype(np.float32),
            'future_obs': goal_obs_t.astype(np.float32),
            'action': self.data_storage['action'][ep_idx, t],
        }
        if self.meta_info['discount'] is not None:
            out['discount'] = self.meta_info['discount']
        return out

