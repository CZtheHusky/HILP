import dataclasses
import typing as tp

import numpy as np
from dm_env import StepType, specs

# Reuse the existing single-environment factory without modifying it
from url_benchmark.dmc_utils.dmc import make as make_single_env


@dataclasses.dataclass
class BatchedTimeStep:
    """Batched version of the TimeStep.

    All fields are numpy arrays stacked over the leading batch dimension.
    physics is kept as a list (one per env) to avoid large copies.
    """
    step_type: np.ndarray  # shape: [num_envs], dtype: int (dm_env.StepType values)
    reward: np.ndarray     # shape: [num_envs], dtype: float32
    discount: np.ndarray   # shape: [num_envs], dtype: float32
    observation: np.ndarray  # shape: [num_envs, ...]
    physics: tp.List[tp.Any]

    def first(self) -> np.ndarray:
        return self.step_type == StepType.FIRST

    def mid(self) -> np.ndarray:
        return self.step_type == StepType.MID

    def last(self) -> np.ndarray:
        return self.step_type == StepType.LAST


class VectorEnvSync:
    """Synchronous vectorized wrapper around multiple DMC envs.

    - Constructs N independent single envs using the existing factory.
    - reset() and step() operate on all envs and return batched timesteps.
    - If an env returns LAST at step, it is auto-reset on the next step call.
    """

    def __init__(
        self,
        name: str,
        num_envs: int,
        obs_type: str = "states",
        frame_stack: int = 1,
        action_repeat: int = 1,
        seed: int = 1,
        image_wh: int = 64,
        seed_offset: int = 1000,
    ) -> None:
        assert num_envs >= 1
        self._name = name
        self._num_envs = num_envs
        self._obs_type = obs_type
        self._envs = [
            make_single_env(
                name=name,
                obs_type=obs_type,
                frame_stack=frame_stack,
                action_repeat=action_repeat,
                seed=seed + i * seed_offset,
                image_wh=image_wh,
            )
            for i in range(num_envs)
        ]

        # Specs are assumed identical across envs; take from the first one
        self._action_spec: specs.Array = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()

        # Track which envs terminated on the last step
        self._needs_reset = np.zeros((num_envs,), dtype=bool)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def action_spec(self) -> specs.Array:
        return self._action_spec

    def observation_spec(self) -> tp.Any:
        return self._observation_spec

    def reset(self) -> BatchedTimeStep:
        timesteps = [env.reset() for env in self._envs]
        return self._stack_timesteps(timesteps)

    def step(self, actions: np.ndarray) -> BatchedTimeStep:
        """Step all environments synchronously.

        actions: np.ndarray with shape [num_envs, action_dim]
        """
        assert actions.shape[0] == self._num_envs, (
            f"actions batch dim {actions.shape[0]} != num_envs {self._num_envs}"
        )

        timesteps = []
        for i, env in enumerate(self._envs):
            if self._needs_reset[i]:
                ts = env.reset()
                self._needs_reset[i] = False
            else:
                ts = env.step(actions[i])
            timesteps.append(ts)

        # Mark envs that ended; will auto-reset on next step()
        for i, ts in enumerate(timesteps):
            if ts.last():
                self._needs_reset[i] = True

        return self._stack_timesteps(timesteps)

    def _stack_timesteps(self, timesteps: tp.List[tp.Any]) -> BatchedTimeStep:
        # Each ts is a url_benchmark.dmc_utils.dmc.TimeStep (or ExtendedTimeStep)
        step_type = np.array([ts.step_type for ts in timesteps])
        reward = np.array([ts.reward for ts in timesteps], dtype=np.float32)
        discount = np.array([ts.discount for ts in timesteps], dtype=np.float32)

        # Observations are arrays already (flattened states or stacked pixels)
        obs = np.stack([ts.observation for ts in timesteps], axis=0)
        physics = [ts.physics for ts in timesteps]
        return BatchedTimeStep(step_type=step_type, reward=reward, discount=discount, observation=obs, physics=physics)


def make_vectorized(
    name: str,
    num_envs: int,
    obs_type: str = "states",
    frame_stack: int = 1,
    action_repeat: int = 1,
    seed: int = 1,
    image_wh: int = 64,
    seed_offset: int = 1000,
) -> VectorEnvSync:
    """Factory for a synchronous vectorized DMC env without touching existing code."""
    return VectorEnvSync(
        name=name,
        num_envs=num_envs,
        obs_type=obs_type,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        image_wh=image_wh,
        seed_offset=seed_offset,
    )


def _self_test() -> None:
    """Run a short smoke test to validate vectorization works end-to-end."""
    try:
        env = make_vectorized(name="walker_run", num_envs=4, obs_type="states", frame_stack=1, action_repeat=1, seed=1)
        ts = env.reset()
        assert ts.observation.shape[0] == env.num_envs
        action_dim = env.action_spec().shape[0]
        for _ in range(5):
            actions = np.random.uniform(low=-1.0, high=1.0, size=(env.num_envs, action_dim)).astype(np.float32)
            ts = env.step(actions)
            # Basic shape checks
            assert ts.observation.shape[0] == env.num_envs
            assert ts.reward.shape == (env.num_envs,)
        print("VectorEnvSync self-test passed.")
    except Exception as e:
        print("VectorEnvSync self-test failed:", repr(e))
        raise


if __name__ == "__main__":
    _self_test()


