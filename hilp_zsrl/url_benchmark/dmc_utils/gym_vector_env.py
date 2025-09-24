import typing as tp

import numpy as np
import gymnasium
# Prefer gymnasium, fallback to gym
from gymnasium import spaces  # type: ignore

from dm_env import specs

from url_benchmark.dmc_utils.dmc import make as make_single_env


def _spec_to_space(spec: specs.Array) -> spaces.Box:
    shape = tuple(int(x) for x in spec.shape)
    dtype = np.dtype(spec.dtype)
    if isinstance(spec, specs.BoundedArray):
        low = np.broadcast_to(np.array(spec.minimum, dtype=dtype), shape)
        high = np.broadcast_to(np.array(spec.maximum, dtype=dtype), shape)
    else:
        # Unbounded: pick a large finite box
        low = np.full(shape, -np.inf, dtype=dtype)
        high = np.full(shape, np.inf, dtype=dtype)
    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


class DMCGymEnv(gymnasium.Env):  # type: ignore
    """Gym/Gymnasium wrapper over existing DMC env factory.

    This does not modify the underlying implementation; it adapts the API so
    gym.vector.AsyncVectorEnv can manage multiple instances asynchronously.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        name: str,
        obs_type: str = "states",
        frame_stack: int = 1,
        action_repeat: int = 1,
        seed: int = 1,
        image_wh: int = 64,
    ) -> None:
        super().__init__()
        self._make_kwargs = dict(
            name=name,
            obs_type=obs_type,
            frame_stack=frame_stack,
            action_repeat=action_repeat,
            seed=seed,
            image_wh=image_wh,
        )
        self._env = make_single_env(**self._make_kwargs)
        # Convert specs
        self.observation_space = _spec_to_space(self._env.observation_spec())
        self.action_space = _spec_to_space(self._env.action_spec())

    def reset(self, *, seed: tp.Optional[int] = None, options: tp.Optional[dict] = None):  # type: ignore
        # We keep the existing env; optional: recreate with a new seed here if needed
        ts = self._env.reset()
        obs = ts.observation
        # include physics snapshot in info so vector env can stack it
        info = {"physics": ts.physics}
        return obs, info

    def step(self, action):  # type: ignore
        ts = self._env.step(action)
        obs = ts.observation
        reward = float(ts.reward)
        terminated = bool(ts.last())
        truncated = False
        # include physics snapshot for each step
        info = {"physics": ts.physics}
        return obs, reward, terminated, truncated, info

    def render(self):  # type: ignore
        try:
            return self._env.render(height=256, width=256, camera_id=0)
        except Exception:
            return None

    def close(self):  # type: ignore
        return None


def make_gym_env_ctor(
    name: str,
    obs_type: str = "states",
    frame_stack: int = 1,
    action_repeat: int = 1,
    seed: int = 1,
    image_wh: int = 64,
):
    def _thunk():
        return DMCGymEnv(
            name=name,
            obs_type=obs_type,
            frame_stack=frame_stack,
            action_repeat=action_repeat,
            seed=seed,
            image_wh=image_wh,
        )
    return _thunk


def make_gym_async_vectorized(
    name: str,
    num_envs: int,
    obs_type: str = "states",
    frame_stack: int = 1,
    action_repeat: int = 1,
    seed: int = 1,
    image_wh: int = 64,
    seed_offset: int = 1000,
):
    """Return gymnasium.vector.AsyncVectorEnv managing DMC envs in subprocesses."""
    ctors = [
        make_gym_env_ctor(
            name=name,
            obs_type=obs_type,
            frame_stack=frame_stack,
            action_repeat=action_repeat,
            seed=seed + i * seed_offset,
            image_wh=image_wh,
        )
        for i in range(num_envs)
    ]
    return gymnasium.vector.AsyncVectorEnv(ctors)  # type: ignore


def _self_test() -> None:
    try:
        venv = make_gym_async_vectorized("walker_run", num_envs=4, obs_type="states")
        obs, infos = venv.reset()
        assert obs.shape[0] == 4
        actions = np.zeros((4,) + venv.single_action_space.shape, dtype=venv.single_action_space.dtype)
        obs, rew, term, trunc, infos = venv.step(actions)
        assert obs.shape[0] == 4 and rew.shape[0] == 4
        venv.close()
        print("gymnasium AsyncVectorEnv self-test passed.")
    except Exception as e:
        print("gymnasium AsyncVectorEnv self-test failed:", repr(e))
        raise


if __name__ == "__main__":
    _self_test()


