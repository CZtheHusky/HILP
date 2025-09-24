# generic_replay.py
import typing as tp
import dataclasses
import collections
import numpy as np
import torch
from url_benchmark.dataset_utils.replay_buffer import DataBuffer
from collections import defaultdict
from tqdm import tqdm
import traceback

# ----------------------------
# 批次容器，与原版兼容
# ----------------------------
T = tp.TypeVar("T", np.ndarray, torch.Tensor)
B = tp.TypeVar("B", bound="EpisodeBatch")

@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None      # 兼容原字段名
    future_obs: tp.Optional[T] = None
    privileged_obs: tp.Optional[T] = None
    commands: tp.Optional[T] = None

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if field.name == "meta":
                out[field.name] = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
            elif isinstance(data, (np.ndarray, torch.Tensor)):
                out[field.name] = torch.as_tensor(data, device=device)
            else:
                out[field.name] = data
        return EpisodeBatch(**out)  # type: ignore

    @classmethod
    def collate_fn(cls, batches: tp.List["EpisodeBatch[T]"]) -> "EpisodeBatch[torch.Tensor]":
        # 若第一项是 numpy，统一转 torch/cpu 再堆叠
        if isinstance(batches[0].obs, np.ndarray):
            batches = [b.to("cpu") for b in batches]  # type: ignore
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(cls):
            data = [getattr(b, field.name) for b in batches]
            if data[0] is None:
                if any(x is not None for x in data):
                    raise RuntimeError("Mixed None and non-None in collate")
                out[field.name] = None
                continue
            if field.name == "meta":
                meta_keys = data[0].keys()
                out[field.name] = {k: torch.stack([d[k] for d in data]) for k in meta_keys}
            elif isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                raise RuntimeError(f"Unsupported field in collate: {field.name}")
        return EpisodeBatch(**out)  # type: ignore

    def unpack(self) -> tp.Tuple[T, T, T, T, T]:
        return (self.obs, self.action, self.reward, self.discount, self.next_obs)

    def with_no_reward(self: B) -> B:
        r = self.reward
        r = torch.zeros_like(r) if isinstance(r, torch.Tensor) else np.zeros_like(r)
        return dataclasses.replace(self, reward=r)



# ============================================================
# Schema & Batch（与原实现兼容/一致）
# ============================================================
@dataclasses.dataclass
class ReplaySchema:
    obs_key: str = "observation"
    action_key: str = "actions"
    reward_key: str = "rewards"
    discount_key: tp.Optional[str] = None
    done_key: tp.Optional[str] = None

    obs_horizon: int = 1
    cast_dtype: tp.Optional[np.dtype] = np.float32
    meta_keys: tp.Optional[tp.Set[str]] = None

    goal_source_key: tp.Optional[str] = None
    is_fixed_episode_length: bool = False

    def core_keys(self) -> tp.Set[str]:
        s = {self.obs_key, self.action_key, self.reward_key}
        if self.discount_key:
            s.add(self.discount_key)
        if self.done_key:
            s.add(self.done_key)
        return s


T = tp.TypeVar("T", np.ndarray, torch.Tensor)
B = tp.TypeVar("B", bound="EpisodeBatch")

@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None
    future_obs: tp.Optional[T] = None
    privileged_obs: tp.Optional[T] = None
    commands: tp.Optional[T] = None

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if field.name == "meta":
                out[field.name] = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
            elif isinstance(data, (np.ndarray, torch.Tensor)):
                out[field.name] = torch.as_tensor(data, device=device)
            else:
                out[field.name] = data
        return EpisodeBatch(**out)  # type: ignore

    @classmethod
    def collate_fn(cls, batches: tp.List["EpisodeBatch[T]"]) -> "EpisodeBatch[torch.Tensor]":
        if isinstance(batches[0].obs, np.ndarray):
            batches = [b.to("cpu") for b in batches]  # type: ignore
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(cls):
            data = [getattr(b, field.name) for b in batches]
            if data[0] is None:
                if any(x is not None for x in data):
                    raise RuntimeError("Mixed None and non-None in collate")
                out[field.name] = None
                continue
            if field.name == "meta":
                meta_keys = data[0].keys()
                out[field.name] = {k: torch.stack([d[k] for d in data]) for k in meta_keys}
            elif isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                raise RuntimeError(f"Unsupported field in collate: {field.name}")
        return EpisodeBatch(**out)  # type: ignore

    def unpack(self) -> tp.Tuple[T, T, T, T, T]:
        return (self.obs, self.action, self.reward, self.discount, self.next_obs)

    def with_no_reward(self: B) -> B:
        r = self.reward
        r = torch.zeros_like(r) if isinstance(r, torch.Tensor) else np.zeros_like(r)
        return dataclasses.replace(self, reward=r)


class StepReplayBuffer:
    def __init__(
        self,
        capacity_steps: int,
        schema: ReplaySchema,
        discount: float = 1.0,
        future: float = 1.0,         # HER: <1 启用，=1 关闭
        p_randomgoal: float = 0.0,   # HER: 随机目标概率
    ):
        assert capacity_steps > 0
        assert 0 <= future <= 1
        assert 0 <= p_randomgoal <= 1

        self.schema = schema
        self.capacity = int(capacity_steps)
        self._global_discount = float(discount)
        self._future = float(future)
        self._p_randomgoal = float(p_randomgoal)
        self._obs_horizon = int(schema.obs_horizon)
        assert self._obs_horizon > 0
        self._meta_store = dict()

        # per-step 存储
        self._storage: tp.Dict[str, np.ndarray] = {}       # core & 非 core（meta 一并放这，按步存）
        # 索引与辅助信息（按步）
        self._valid = np.zeros(self.capacity, dtype=bool)
        self._ep_id = np.full(self.capacity, -1, dtype=np.int32)     # episode id（单调递增）
        self._step_no = np.zeros(self.capacity, dtype=np.int32)      # 该步在其 episode 内的索引
        self._ep_last_step_no = np.zeros(self.capacity, dtype=np.int32) # 该步所在 episode 的最后一步episode内索引（T-1）
        self._abs_ep_last_step = np.zeros(self.capacity, dtype=np.int32) # 该步所在 episode 的最后一步绝对索引（T-1）
        self._core_key = set(self.schema.core_keys())

        # ring 指针与统计
        self._w = 0
        self._size = 0
        self._episode_counter = 0


    def _maybe_cast(self, arr: tp.Union[np.ndarray, float, int, bool]) -> np.ndarray:
        if np.isscalar(arr):
            arr = np.array([arr])
        arr = np.asarray(arr)
        if self.schema.cast_dtype is not None and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(self.schema.cast_dtype, copy=False)
        return arr

    def add_episode(self, episode_data: tp.Dict[str, np.ndarray], meta: tp.Dict[str, np.ndarray]) -> None:
        """一次性写入整条 episode。"""
        key_len = [v.shape[0] for v in episode_data.values()]
        assert len(set(key_len)) == 1, "All keys must have the same length"
        assert key_len[0] <= self.capacity, "Episode length exceeds capacity"
        episode_len = key_len[0]
        # episode id
        ep_id = self._episode_counter
        self._episode_counter += 1
        self._episode_counter = self._episode_counter % self.capacity
        if episode_len + self._w > self.capacity:
            self._size = self._w
            self._w = 0 # reset write pointer, it's ok to leave several blanks at the last of the buffer
        else:
            self._size = max(episode_len + self._w, self._size)
        index_start = self._w
        index_end = episode_len + self._w
        ep_ids = set(self._ep_id[index_start:index_end].tolist())
        if index_end < self.capacity:
            overwrited_ep_id = self._ep_id[index_end]
            current_abs_start = index_end
            current_abs_end = self._abs_ep_last_step[index_end]
            if current_abs_end > current_abs_start:
                self._ep_last_step_no[current_abs_start:current_abs_end] -= self._step_no[current_abs_start]
                self._step_no[current_abs_start:current_abs_end] -= self._step_no[current_abs_start]
            if overwrited_ep_id in ep_ids:
                ep_ids.remove(overwrited_ep_id)
        for ep_iid in ep_ids:
            self._meta_store.pop(ep_iid, None)
        self._ep_id[index_start:index_end] = ep_id
        self._step_no[index_start:index_end] = np.arange(episode_len)
        self._ep_last_step_no[index_start:index_end] = episode_len - 1
        self._abs_ep_last_step[index_start:index_end] = index_end
        self._valid[index_start:index_end] = True
        for key, values in episode_data.items():
            if key in self._core_key:
                values = self._maybe_cast(values)
                if key not in self._storage:
                    self._storage[key] = np.empty((self.capacity,) + values.shape[1:], dtype=values.dtype)
                self._storage[key][index_start:index_end] = values
        self._w = index_end
        self._meta_store[ep_id] = meta


 # ---------------- 采样相关 ----------------
    def __len__(self) -> int:
        return self._size  # 存的 step 数

    @property
    def num_steps(self) -> int:
        return self._size

    @property
    def num_episodes_seen(self) -> int:
        return len(self._meta_store)

    def _candidate_indices(self) -> np.ndarray:
        """
        可采样的 step i 需满足：
          1) i 和 i+1 都有效，且属于同一 episode，step 连续；
          2) obs_horizon 窗口内的 (H-1) 个历史步都有效且属于同一 episode 且 step 连续。
        """
        if self._size == 0:
            return np.empty(0, dtype=np.int64)

        cap = self.capacity
        idx = np.arange(cap, dtype=np.int64)
        idx_next = (idx + 1) % cap

        mask = (
            self._valid
            & self._valid[idx_next]
            & (self._ep_id == self._ep_id[idx_next])
            & (self._step_no[idx_next] == (self._step_no + 1))
        )
        cand = np.nonzero(mask)[0]
        self.cand = cand

    def _gather_obs_window(self, positions: np.ndarray) -> np.ndarray:
        """
        从给定 step 位置收集 obs_horizon 窗口：
            [t-H+1, ..., t]  -> shape: (B, H, *obs_payload)
        若 H=1 则返回 (B, *obs_payload)
        """
        obs_key = self.schema.obs_key
        obs_arr = self._storage[obs_key]
        H = self._obs_horizon
        if H == 1:
            return obs_arr[positions]

        steps = self._step_no[positions]    # (B,)  t 在 episode 内的相对步号 s
        t0 = (positions - steps)[:, None]   # (B,1) episode 绝对首步位置
        offsets = np.arange(-(H - 1), 1,)   # (H,)  [-H+1, ..., 0]

        # 目标绝对索引 = t0 + max(s + offsets, 0)
        # 等价于 clamp(t + offsets, lower=t0)
        idx2d = t0 + np.maximum(steps[:, None] + offsets[None, :], 0)
        return obs_arr[idx2d]

    def sample(self, batch_size: int) -> EpisodeBatch:
        replace = (self.cand.size < batch_size)
        idx = np.random.choice(self.cand, size=batch_size, replace=replace)
        idx_next = idx + 1

        # obs & next_obs（考虑 horizon）
        obs = self._gather_obs_window(idx)
        next_obs = self._gather_obs_window(idx_next)

        schema = self.schema
        action = self._storage[schema.action_key][idx]
        reward = self._storage[schema.reward_key][idx]

        if schema.discount_key and schema.discount_key in self._storage:
            discount_local = self._storage[schema.discount_key][idx]
        else:
            discount_local = np.ones_like(reward, dtype=self.schema.cast_dtype or np.float32)
        discount = self._global_discount * discount_local
        # HER: future_obs
        future_obs = None
        if self._future < 1.0:
            B = idx.shape[0]
            # 计算每个样本的未来 step 位置（同一 episode 内）
            step_now = self._step_no[idx]
            step_last = self._ep_last_step_no[idx]
            # Geometric(p=1-future)
            offsets = np.random.geometric(p=(1.0 - self._future), size=B)
            target_step = np.minimum(step_now + offsets, step_last)
            delta = target_step - step_now  # >= 1
            j = idx + delta
            try:
                future_obs = self._gather_obs_window(j) 
            except Exception as e:
                print(f"Error in gathering future obs: {e}")
                print(f"idx: {idx}, step_now: {step_now}, step_last: {step_last}, offsets: {offsets}, target_step: {target_step}, delta: {delta}, j: {j}")
                print("Traceback:")
                traceback.print_exc()
                raise e

            # 随机目标（跨 episode）
            if self._p_randomgoal > 0.0:
                mask = (np.random.rand(B) < self._p_randomgoal)
                if mask.any():
                    rand_idx = np.random.choice(self.cand, size=int(mask.sum()), replace=True)
                    future_obs_rand = self._gather_obs_window(rand_idx)
                    future_obs[mask] = future_obs_rand

        return EpisodeBatch(
            obs=obs,
            action=action,
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            future_obs=future_obs,
        )

    def load_from_rb(self, rb_path):
        rb = DataBuffer.create_from_path(rb_path)
        episode_ends = rb.meta['episode_ends'][:]
        print(f"Loading {len(episode_ends)} episodes from {rb_path}")
        data_dict = {}

        for key in rb.data.keys():
            if key in self._core_key:
                data_dict[key] = rb.data[key][:]

        for ep_idx in tqdm(range(len(episode_ends))):
            ep_start = 0 if ep_idx == 0 else episode_ends[ep_idx-1]
            ep_end = episode_ends[ep_idx]
            episode_data = dict()
            for key in data_dict.keys():
                episode_data[key] = data_dict[key][ep_start:ep_end]
            meta = dict()
            # for key in rb.meta.keys():
            #     meta[key] = rb.meta[key][ep_start:ep_end]
            self.add_episode(episode_data, meta)
        self._candidate_indices()
        return self

    # ---- 对外 API ----
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "StepReplayBuffer":
        state = torch.load(path, map_location="cpu")
        schema = ReplaySchema(**state['schema'])
        buf = cls(
            capacity_steps=int(state['capacity']),
            schema=schema,
            discount=float(state.get('discount', 1.0)),
            future=float(state.get('future', 1.0)),
            p_randomgoal=float(state.get('p_randomgoal', 0.0)),
        )
        buf.load_state_dict(state)
        return buf

    # ---- 保存/加载的状态 ----
    def state_dict(self):
        to_t = lambda x: torch.from_numpy(x.copy())
        return dict(
            capacity=self.capacity,
            size=int(self._size),
            w=int(self._w),
            discount=float(self._global_discount),
            future=float(self._future),
            p_randomgoal=float(self._p_randomgoal),
            schema=dataclasses.asdict(self.schema),
            storage={k: to_t(v[:self._size]) for k, v in self._storage.items()},  # 只存有效区间
            valid=to_t(self._valid[:self._size]),
            ep_id=to_t(self._ep_id[:self._size]),
            step_no=to_t(self._step_no[:self._size]),
            ep_last_step_no=to_t(self._ep_last_step_no[:self._size]),
            abs_ep_last_step=to_t(self._abs_ep_last_step[:self._size]),
            episode_counter=int(self._episode_counter),
            meta_store=self._meta_store,   # 如过大可精简或置空
        )

    def load_state_dict(self, state):
        to_np = lambda t: t.detach().cpu().numpy()
        self.capacity = int(state['capacity'])
        self._size    = int(state['size'])
        self._w       = int(state['w'])
        self._global_discount = float(state.get('discount', 1.0))
        self._future         = float(state.get('future', 1.0))
        self._p_randomgoal   = float(state.get('p_randomgoal', 0.0))

        # 可选：重建 schema
        if 'schema' in state:
            self.schema = ReplaySchema(**state['schema'])
            self._obs_horizon = int(self.schema.obs_horizon)

        # 重新分配缓冲区（满 capacity），并填充前 size 段
        self._valid = np.zeros(self.capacity, dtype=bool); self._valid[:self._size] = to_np(state['valid']).astype(bool)
        self._ep_id = np.full(self.capacity, -1, dtype=np.int32); self._ep_id[:self._size] = to_np(state['ep_id']).astype(np.int32)
        self._step_no = np.zeros(self.capacity, dtype=np.int32); self._step_no[:self._size] = to_np(state['step_no']).astype(np.int32)
        self._ep_last_step_no = np.zeros(self.capacity, dtype=np.int32); self._ep_last_step_no[:self._size] = to_np(state['ep_last_step_no']).astype(np.int32)
        self._abs_ep_last_step = np.zeros(self.capacity, dtype=np.int32); self._abs_ep_last_step[:self._size] = to_np(state['abs_ep_last_step']).astype(np.int32)

        self._storage = {}
        for k, v in state['storage'].items():
            v_np = to_np(v)
            arr = np.empty((self.capacity,) + v_np.shape[1:], dtype=v_np.dtype)
            arr[:self._size] = v_np
            self._storage[k] = arr

        self._meta_store = state.get('meta_store', {})
        self._candidate_indices()