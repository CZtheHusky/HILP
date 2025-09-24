import dataclasses
import multiprocessing as mp
import typing as tp

import numpy as np
from dm_env import StepType, specs

from url_benchmark.dmc_utils.dmc import make as make_single_env


# Messages exchanged with workers
_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"


def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, make_kwargs: dict) -> None:
    parent_remote.close()
    env = make_single_env(**make_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == _CMD_RESET:
                ts = env.reset()
                remote.send((ts.step_type, ts.reward, ts.discount, ts.observation, ts.physics))
            elif cmd == _CMD_STEP:
                ts = env.step(data)
                remote.send((ts.step_type, ts.reward, ts.discount, ts.observation, ts.physics))
            elif cmd == _CMD_CLOSE:
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except KeyboardInterrupt:
        pass


@dataclasses.dataclass
class AsyncBatchedTimeStep:
    step_type: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    observation: np.ndarray
    physics: tp.List[tp.Any]

    def first(self) -> np.ndarray:
        return self.step_type == StepType.FIRST

    def last(self) -> np.ndarray:
        return self.step_type == StepType.LAST


class AsyncVectorEnv:
    """Asynchronous vectorized env using multiprocessing.

    step_async(actions) sends actions to all workers and returns immediately.
    step_wait() collects results when ready.
    reset() is synchronous for simplicity (can be made async similarly).
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
        start_method: str = "forkserver",
    ) -> None:
        assert num_envs >= 1
        ctx = mp.get_context(start_method)
        self._num_envs = num_envs
        self._remotes, self._work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self._processes: tp.List[mp.Process] = []
        for i in range(num_envs):
            make_kwargs = dict(
                name=name,
                obs_type=obs_type,
                frame_stack=frame_stack,
                action_repeat=action_repeat,
                seed=seed + i * seed_offset,
                image_wh=image_wh,
            )
            p = ctx.Process(target=_worker, args=(self._work_remotes[i], self._remotes[i], make_kwargs))
            p.daemon = True
            p.start()
            self._work_remotes[i].close()
            self._processes.append(p)

        # Query action/observation specs by doing a single reset and inspecting shapes
        ts = self.reset()
        self._last_obs_shape = ts.observation.shape[1:]  # [num_envs, ...]
        # For action_spec, instantiate a temp env in the main process (cheap)
        tmp_env = make_single_env(name=name, obs_type=obs_type, frame_stack=frame_stack, action_repeat=action_repeat, seed=seed, image_wh=image_wh)
        self._action_spec: specs.Array = tmp_env.action_spec()

        self._waiting = False

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def action_spec(self) -> specs.Array:
        return self._action_spec

    def reset(self) -> AsyncBatchedTimeStep:
        for remote in self._remotes:
            remote.send((_CMD_RESET, None))
        outs = [remote.recv() for remote in self._remotes]
        return self._stack(outs)

    def step_async(self, actions: np.ndarray) -> None:
        assert actions.shape[0] == self._num_envs
        for i, remote in enumerate(self._remotes):
            remote.send((_CMD_STEP, actions[i]))
        self._waiting = True

    def step_wait(self) -> AsyncBatchedTimeStep:
        assert self._waiting, "step_async must be called before step_wait"
        outs = [remote.recv() for remote in self._remotes]
        self._waiting = False
        return self._stack(outs)

    def close(self) -> None:
        for remote in self._remotes:
            remote.send((_CMD_CLOSE, None))
        for p in self._processes:
            p.join(timeout=1.0)

    def _stack(self, outs: tp.List[tp.Tuple]) -> AsyncBatchedTimeStep:
        step_type, reward, discount, obs, phy = zip(*outs)
        step_type = np.array(step_type)
        reward = np.array(reward, dtype=np.float32)
        discount = np.array(discount, dtype=np.float32)
        obs = np.stack(obs, axis=0)
        physics = list(phy)
        return AsyncBatchedTimeStep(step_type=step_type, reward=reward, discount=discount, observation=obs, physics=physics)


def make_async_vectorized(
    name: str,
    num_envs: int,
    obs_type: str = "states",
    frame_stack: int = 1,
    action_repeat: int = 1,
    seed: int = 1,
    image_wh: int = 64,
    seed_offset: int = 1000,
    start_method: str = "forkserver",
) -> AsyncVectorEnv:
    return AsyncVectorEnv(
        name=name,
        num_envs=num_envs,
        obs_type=obs_type,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        image_wh=image_wh,
        seed_offset=seed_offset,
        start_method=start_method,
    )


def _self_test() -> None:
    try:
        env = make_async_vectorized(name="walker_run", num_envs=4, obs_type="states", frame_stack=1, action_repeat=1, seed=1)
        ts = env.reset()
        action_dim = env.action_spec().shape[0]
        for _ in range(5):
            actions = np.random.uniform(low=-1.0, high=1.0, size=(env.num_envs, action_dim)).astype(np.float32)
            env.step_async(actions)
            ts = env.step_wait()
            assert ts.observation.shape[0] == env.num_envs
        env.close()
        print("AsyncVectorEnv self-test passed.")
    except Exception as e:
        print("AsyncVectorEnv self-test failed:", repr(e))
        raise


if __name__ == "__main__":
    _self_test()


