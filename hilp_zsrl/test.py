
import platform
import os
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


from url_benchmark.dmc_utils.gym_vector_env import make_gym_async_vectorized
from url_benchmark.dmc_utils.dmc import make as make_single_env


# if __name__ == "__main__":

# env = make_single_env(name="walker_run", obs_type="states", frame_stack=1, action_repeat=1, seed=0)

# if hasattr(env, 'physics'):
#     frame = env.physics.render(height=96, width=96, camera_id=0)
# else:
#     frame = env.base_env.render()
# print(frame.shape)
# print(frame.dtype)
# print(frame)
# env.close()

env = make_gym_async_vectorized(
    name="walker_run",
    num_envs=2,
    obs_type="states",
    frame_stack=1,
    action_repeat=1,
    seed=0,
    image_wh=64,
)

obs, info = env.reset()
# env.step(env.action_space.sample())

frames = env.call('render')
print(frames)
print(len(frames))
print(frames[0].shape)

