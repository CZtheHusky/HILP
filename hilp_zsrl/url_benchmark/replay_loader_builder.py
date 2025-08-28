from pathlib import Path
from typing import Optional

import torch

from url_benchmark.in_memory_replay_buffer import ReplayBuffer


def _load_replay_loader_checkpoint(
    fp: str,
    *,
    num_episodes: Optional[int] = None,
    use_pixels: bool = False,
    discount: float,
    future: float,
) -> ReplayBuffer:
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
        rb = payload
    else:
        # expect dict with key 'replay_loader'
        rb = payload['replay_loader']

    assert isinstance(rb, ReplayBuffer)
    rb._current_episode.clear()  # make sure we can start over
    rb._future = future
    rb._discount = discount
    rb._max_episodes = len(rb._storage["discount"])
    return rb


def build_replay_loader(
    *,
    load_replay_buffer: str,
    replay_buffer_episodes: int,
    obs_type: str,
    frame_stack: Optional[int],
    discount: float,
    future: float,
    p_currgoal: float,
    p_randomgoal: float,
) -> ReplayBuffer:
    """Construct and configure ReplayBuffer exactly as in train_offline.py,
    using explicit parameters instead of a cfg object.

    Sequence:
      1) Load checkpoint (only replay_loader), with optional truncation and pixel remap
      2) Set sampling params and derived fields
    """
    print("loading Replay from %s", load_replay_buffer)
    rb = _load_replay_loader_checkpoint(
        load_replay_buffer,
        num_episodes=replay_buffer_episodes,
        use_pixels=(obs_type == 'pixels'),
        discount=discount,
        future=future,
    )

    # Align runtime sampling params
    rb._future = future
    rb._discount = discount
    rb._p_currgoal = p_currgoal
    rb._p_randomgoal = p_randomgoal
    rb._frame_stack = frame_stack if obs_type == 'pixels' else None
    rb._max_episodes = len(rb._storage["discount"])
    return rb


