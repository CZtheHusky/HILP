# https://github.com/real-stanford/diffusion_policy/

from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property
from tqdm import tqdm

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=1024 * 1024 * 2, max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if this_chunk_bytes <= target_chunk_bytes and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class DataBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimention to be time. Only chunk in time dimension.
    """

    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert "data" in root
        assert "meta" in root
        assert "episode_ends" in root["meta"]
        for key, value in root["data"].items():
            assert (
                value.shape[0] == root["meta"]["episode_ends"][-1]
            ), f"{key} {value.shape} {root['meta']['episode_ends'].shape} {root['meta']['episode_ends'][-1]}"
            if not (
                value.shape[0] == root["meta"]["episode_ends"][-1]
            ):
                print(f"{key} {value.shape} {root['meta']['episode_ends'].shape} {root['meta']['episode_ends'][-1]}")
        self.root = root
        
    def truncate_data(self, key, target_size):
        if len(self.data[key].shape) == 1:
            self.data[key].resize((target_size,))
        else:
            self.data[key].resize((target_size, ) + self.data[key].shape[1:])
        
    def truncate_meta(self, key, target_size):
        if len(self.meta[key].shape) == 1:
            self.meta[key].resize((target_size,))
        else:
            self.meta[key].resize((target_size, ) + self.meta[key].shape[1:])

    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group("data", overwrite=False)
        meta = root.require_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            episode_ends = meta.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None, overwrite=False)
        return cls(root=root)

    def new_meta_key(self, key, shape=(0,), dtype=np.int64, compressor=None, overwrite=False):
        if key not in self.meta:
            self.meta.zeros(name=key, shape=shape, dtype=dtype, compressor=compressor, overwrite=overwrite)

    @classmethod
    def create_empty_numpy(cls):
        root = {"data": dict(), "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)}}
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        if "data" not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode="r", **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)

    @classmethod
    def copy_from_store(
        cls,
        src_store,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
        if_exists="replace",
        num_episodes=None,  # 新增参数：指定要读取的 episode 数量
        world_size=None,
        rank=None,
        **kwargs,
    ):
        """
        Load a specific number of episodes into memory.
        """
        src_root = zarr.group(src_store)
        root = None
        n_eps = src_root["meta"]["episode_ends"].shape[0]
        if world_size is None:
            assert rank is None
            world_size = 1
            rank = 0
        n_eps_per_process = math.ceil(n_eps / world_size)
        start_ep_id = rank * n_eps_per_process
        if rank == world_size - 1:
            end_ep_id = n_eps
        else:
            end_ep_id = min(start_ep_id + n_eps_per_process, n_eps)
        
            
            
        # 获取 episode_ends 数组
        # start_episode = 0
        episode_ends = src_root["meta"]["episode_ends"][start_ep_id:end_ep_id]
        
        # if num_episodes is not None:
        #     print(f"Loading {num_episodes} episodes")
        # else:
        #     num_episodes = len(episode_ends)
        # num_episodes = min(num_episodes, len(episode_ends))
        if rank == 0:
            start_idx = 0
        else:
            start_idx = src_root["meta"]["episode_ends"][start_ep_id-1]
        # episode_ends = episode_ends[:num_episodes]
        end_idx = episode_ends[-1]
        
        if store is None:
            meta = dict()
            print("Loading metadata:")
            for key, value in tqdm(src_root["meta"].items(), desc="Metadata"):
                if key != "episode_ends":
                    if len(value.shape) == 0:
                        meta[key] = np.array(value)[start_ep_id:end_ep_id]
                    else:
                        meta[key] = value[start_ep_id:end_ep_id]

            if keys is None:
                keys = src_root["data"].keys()
            data = dict()
            print("Loading data arrays:")
            for key in tqdm(keys, desc="Data arrays"):
                arr = src_root["data"][key]
                data[key] = arr[start_idx:end_idx]
            if start_ep_id > 0:
                episode_ends = episode_ends - src_root["meta"]["episode_ends"][start_ep_id - 1]
            meta["episode_ends"] = episode_ends
            root = {"meta": meta, "data": data}
        else:
            raise NotImplementedError
            root = zarr.group(store)
            print("Copying metadata:")
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=src_store, dest=store, source_path="/meta", dest_path="/meta", if_exists=if_exists
            )
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            print("Copying data arrays:")
            for key in tqdm(keys, desc="Data arrays"):
                value = src_root["data"][key]
                data_range = value[start_idx:end_idx]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store, source_path=this_path, dest_path=this_path, if_exists=if_exists
                    )
                else:
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=data_range, dest=data_group, name=key, chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer


    @classmethod
    def copy_from_path(
        cls,
        zarr_path,
        backend=None,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
        if_exists="replace",
        num_episodes=None,  # 新增参数：指定要读取的 episode 数量
        world_size=None,
        rank=None,
        **kwargs,
    ):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == "numpy":
            print("backend argument is depreacted!")
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), "r")
        for key in group["data"].keys():
            print(f"{key} chunks: {group['data'][key].chunks}")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            num_episodes=num_episodes,
            world_size=world_size,
            rank=rank,
            **kwargs,
        )

    # ============= save methods ===============
    def save_to_store(
        self,
        store,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):

        root = zarr.group(store)
        if self.backend == "zarr":
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store, source_path="/meta", dest_path="/meta", if_exists=if_exists
            )
        else:
            meta_group = root.create_group("meta", overwrite=True)
            # save meta, no chunking
            for key, value in self.root["meta"].items():
                _ = meta_group.array(name=key, data=value, shape=value.shape, chunks=value.shape)

        # save data, chunk
        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key, chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(name=key, data=value, chunks=cks, compressor=cpr)
        return store

    def save_to_path(
        self,
        zarr_path,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor="default"):
        if compressor == "default":
            compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == "disk":
            compressor = numcodecs.Blosc("zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = "nil"
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == "nil":
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: Union[dict, tuple], key, array, target_chunk_bytes=1024 * 1024 * 2):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype, target_chunk_bytes=target_chunk_bytes)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                _ = meta_group.array(name=key, data=value, shape=value.shape, chunks=value.shape, overwrite=True)
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    def get_episode_idxs(self):
        import numba

        numba.jit(nopython=True)

        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    @property
    def backend(self):
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == "zarr":
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
        
    def add_chunked_meta(
        self,
        data: Dict[str, np.ndarray],
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        target_chunk_bytes=1024 * 1024 * 2,
    ):
        is_zarr = self.backend == "zarr"
        for key, value in data.items():
            if key not in self.meta:
                current_len = 0
            else:
                current_len = self.meta[key].shape[0]
            new_shape = (current_len + value.shape[0],) + value.shape[1:]
            # create array
            if key not in self.meta:
                if is_zarr:
                    cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value, target_chunk_bytes=target_chunk_bytes)
                    cpr = self._resolve_array_compressor(compressors=compressors, key=key, array=value)
                    arr = self.meta.zeros(name=key, shape=new_shape, chunks=cks, dtype=value.dtype, compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.meta[key] = arr
            else:
                arr = self.meta[key]
                assert value.shape[1:] == arr.shape[1:], f"key {key} shape issue, {value.shape[1:]} != {arr.shape[1:]}"
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy meta
            arr[-value.shape[0] :] = value
            
    def add_chunked_data(
        self,
        data: Dict[str, np.ndarray],
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        target_chunk_bytes=1024 * 1024 * 2,
    ):
        is_zarr = self.backend == "zarr"
        for key, value in data.items():
            if key not in self.data:
                current_len = 0
            else:
                current_len = self.data[key].shape[0]
            new_shape = (current_len + value.shape[0],) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value, target_chunk_bytes=target_chunk_bytes)
                    cpr = self._resolve_array_compressor(compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, shape=new_shape, chunks=cks, dtype=value.dtype, compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:], f"key {key} shape issue, {value.shape[1:]} != {arr.shape[1:]}"
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy meta
            arr[-value.shape[0] :] = value


    def add_episode(
        self,
        data: Dict[str, np.ndarray],
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
    ):
        assert len(data) > 0
        is_zarr = self.backend == "zarr"

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert len(value.shape) >= 1
            if episode_length is None:
                episode_length = len(value)
            else:
                assert episode_length == len(value)
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, shape=new_shape, chunks=cks, dtype=value.dtype, compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:], f"key {key} shape issue: {value.shape} {arr.shape}"
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0] :] = value

        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, "episode_ends", chunk_length=int(episode_ends.shape[0] * 1.5))
    
    def add_meta(self, key: str, value: np.ndarray):
        """
        Add metadata key-value pair to the ReplayBuffer's meta group and save it to disk.
        
        Parameters:
            key (str): The metadata key to add.
            value (np.ndarray): The metadata value, must be a NumPy array.
        """
        # Ensure the value is a NumPy array
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        # Check for incompatible types
        if value.dtype == object:
            raise TypeError(f"Invalid value type for key '{key}': {value.dtype}. Must not be object type.")

        # Add the key-value pair to meta
        if self.backend == "zarr":
            # Zarr backend: Add to Zarr group and save to disk
            meta_group = self.meta
            if key in meta_group:
                # Overwrite existing key
                meta_group[key][:] = value
            else:
                # Create new key in the Zarr meta group
                meta_group.array(name=key, data=value, shape=value.shape, chunks=value.shape, overwrite=True)
        else:
            # Numpy backend: Directly update the in-memory dictionary
            self.meta[key] = value


    def drop_episode(self):
        is_zarr = self.backend == "zarr"
        episode_ends = self.episode_ends[:].copy()
        assert len(episode_ends) > 0
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def pop_episode(self):
        assert self.n_episodes > 0
        episode = self.get_episode(self.n_episodes - 1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result

    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == "zarr"
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict):
        assert self.backend == "zarr"
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == "zarr"
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == "zarr"
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
