import os
from typing import Optional

import torch

from url_benchmark.in_memory_replay_buffer import ReplayBuffer


def load_replay_buffer_from_checkpoint(fp: str) -> ReplayBuffer:
    payload = torch.load(fp, map_location='cpu')
    if isinstance(payload, ReplayBuffer):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get('replay_loader', None), ReplayBuffer):
        return payload['replay_loader']
    raise RuntimeError(f"Unsupported checkpoint format at {fp}")


def configure_replay_buffer(
    rb: ReplayBuffer,
    *,
    discount: float,
    future: float,
    p_randomgoal: float,
    frame_stack: Optional[int] = None,
    use_pixels: bool = False,
) -> ReplayBuffer:
    # align internal sampling params with offline training pipeline
    rb._future = future
    rb._discount = discount
    rb._p_currgoal = 0.0  # ZSRL style: merge into random goal mix
    rb._p_randomgoal = p_randomgoal
    if frame_stack is not None:
        rb._frame_stack = frame_stack

    # optional pixel → observation remap
    if use_pixels and 'pixel' in rb._storage and 'observation' not in rb._storage:
        rb._storage['observation'] = rb._storage['pixel']
        try:
            del rb._storage['pixel']
            if 'pixel' in rb._batch_names:
                rb._batch_names.remove('pixel')
        except Exception:
            pass

    # refresh capacity derived fields
    rb._max_episodes = len(rb._storage.get('discount', [])) or rb._max_episodes
    return rb


