import time
import numpy as np

from url_benchmark.dmc_utils.dmc import make as make_single_env
from url_benchmark.dmc_utils.gym_vector_env import make_gym_async_vectorized


def benchmark_single(name: str, total_steps: int, obs_type: str = "states"):
    env = make_single_env(name=name, obs_type=obs_type, frame_stack=1, action_repeat=1, seed=1)
    ts = env.reset()
    action_spec = env.action_spec()
    low = np.broadcast_to(action_spec.minimum, action_spec.shape).astype(np.float32)
    high = np.broadcast_to(action_spec.maximum, action_spec.shape).astype(np.float32)
    t0 = time.time()
    steps = 0
    while steps < total_steps:
        action = np.random.uniform(low=low, high=high).astype(np.float32)
        ts = env.step(action)
        if ts.last():
            ts = env.reset()
        steps += 1
    dt = time.time() - t0
    env_fps = total_steps / dt
    return env_fps, dt


def benchmark_gym_async(name: str, num_envs: int, steps_per_env: int, obs_type: str = "states"):
    venv = make_gym_async_vectorized(name=name, num_envs=num_envs, obs_type=obs_type)
    obs, infos = venv.reset()
    act_shape = venv.single_action_space.shape
    low = np.broadcast_to(venv.single_action_space.low, act_shape).astype(np.float32)
    high = np.broadcast_to(venv.single_action_space.high, act_shape).astype(np.float32)

    t0 = time.time()
    for _ in range(steps_per_env):
        actions = np.random.uniform(low=low, high=high, size=(num_envs,) + act_shape).astype(np.float32)
        obs, rew, term, trunc, infos = venv.step(actions)
        done = term | trunc
        if done.any():
            assert done.all()
            print(_ + 1)
        # Gymnasium AsyncVectorEnv typically auto-resets done envs; no manual reset here
    dt = time.time() - t0
    total_steps = num_envs * steps_per_env
    env_fps = total_steps / dt
    venv.close()
    return env_fps, dt


if __name__ == "__main__":
    task = "walker_run"
    num_envs = 100
    steps_per_env = 10000
    total_single_steps = num_envs * steps_per_env

    # single_fps, single_dt = benchmark_single(task, total_single_steps, obs_type="states")
    async_fps, async_dt = benchmark_gym_async(task, num_envs, steps_per_env, obs_type="states")

    # print(f"Single-env: {total_single_steps} steps in {single_dt:.2f}s -> {single_fps:.1f} steps/s")
    print(f"Gym Async ({num_envs} envs): {num_envs*steps_per_env} steps in {async_dt:.2f}s -> {async_fps:.1f} steps/s")
    # print(f"Speedup (async / single): {async_fps / single_fps:.2f}x")