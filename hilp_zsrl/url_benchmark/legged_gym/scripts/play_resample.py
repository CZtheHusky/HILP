import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym import gymapi
import numpy as np
import torch
import tqdm


def play(args):
    # Load env and train cfgs for the specified task
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Visualization-friendly overrides
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 100000  # long episode for continuous viewing

    # Ensure commands are resampled every 10 seconds using the env's training sampler
    env_cfg.commands.resampling_time = 10

    # Disable terrain curriculum so periodic command resampling triggers uniformly
    # (training sampler is still used internally by the env)
    env_cfg.terrain.curriculum = False

    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load policy for inference
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Camera setup (simple tracking of the first env)
    track_index = 0
    look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
    camera_relative_position = np.array([2.0, 0.0, 0.8])
    env.set_camera(look_at + camera_relative_position, look_at, track_index)

    # Reset and take one zero-action step to fetch initial observations
    _, _ = env.reset()
    obs, critic_obs, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device))

    # Run loop
    timesteps = int(env_cfg.env.episode_length_s / max(env.dt, 1e-6))
    for _ in tqdm.tqdm(range(timesteps)):
        with torch.inference_mode():
            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
            obs, critic_obs, _, _, _ = env.step(actions)

            # Update simple follow camera
            look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
            env.set_camera(look_at + camera_relative_position, look_at, track_index)


if __name__ == '__main__':
    args = get_args()
    play(args)


