import os
import sys

# ensure isaacgym is imported before torch
import isaacgym  # noqa: F401

sys.path.append(os.getcwd())
from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
import torch
import numpy as np
from isaacgym import gymapi


class OnPolicyTestRunner:
    def __init__(self, task: str, num_envs: int, headless: bool = True, load_checkpoint: str = None,
                 record_video: bool = False, video_dir: str = "videos/on_policy_test",
                 camera_width: int = 640, camera_height: int = 480, frame_stride: int = 10) -> None:
        from isaacgym import gymutil
        args = gymutil.parse_arguments(description="OnPolicyTestRunner (no learning)", custom_parameters=[
            {"name": "--task", "type": str, "default": task},
            {"name": "--num_envs", "type": int, "default": num_envs},
            {"name": "--headless", "action": "store_true", "default": headless},
            {"name": "--episodes_per_env", "type": int, "default": 1},
            {"name": "--load_checkpoint", "type": str},
            {"name": "--record_video", "action": "store_true", "default": record_video},
            {"name": "--video_dir", "type": str, "default": video_dir},
            {"name": "--camera_width", "type": int, "default": camera_width},
            {"name": "--camera_height", "type": int, "default": camera_height},
            {"name": "--frame_stride", "type": int, "default": frame_stride},
        ])
        args.headless = headless or getattr(args, "headless", False)
        self._record_video = getattr(args, "record_video", record_video)
        self._video_dir = getattr(args, "video_dir", video_dir)
        self._camera_width = getattr(args, "camera_width", camera_width)
        self._camera_height = getattr(args, "camera_height", camera_height)
        self._frame_stride = max(1, getattr(args, "frame_stride", frame_stride))
        # If viewer enabled, force single env for better camera control
        if not args.headless:
            args.num_envs = min(num_envs, 1)
        else:
            args.num_envs = num_envs

        # make env, and enforce 10s episode to match collector
        self.env, self.env_cfg = task_registry.make_env(name=task, args=args)
        try:
            self.env_cfg.env.episode_length_s = 10.0
        except Exception:
            pass
        self.device = self.env.device

        # optional policy
        self.policy = None
        if load_checkpoint:
            try:
                env = self.env
                train_runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args)
                # if a path is given, try use it
                if os.path.isfile(load_checkpoint):
                    train_runner.load(load_checkpoint)
                self.policy = train_runner.get_inference_policy(device=self.device)
            except Exception:
                self.policy = None

        _, _ = self.env.reset()

        # viewer camera setup when not headless
        self._use_viewer = not args.headless
        if self._use_viewer:
            track_index = 0
            look_at = np.array(self.env.root_states[track_index, :3].cpu(), dtype=np.float64)
            camera_relative_position = np.array([1.5, 0.0, 1.0])
            self.env.set_camera(look_at + camera_relative_position, look_at, track_index)
            # camera motion state
            self._camera_rot = np.pi * 8 / 10
            self._camera_rot_per_sec = 1 * np.pi / 10
            self._camera_h_scale = 1.0
            self._camera_v_scale = 0.8
            self._track_index = track_index

        # sensor camera for recording (works in headless or viewer modes)
        self._camera_handle = None
        if self._record_video:
            os.makedirs(self._video_dir, exist_ok=True)
            cam_props = gymapi.CameraProperties()
            cam_props.width = self._camera_width
            cam_props.height = self._camera_height
            self._camera_handle = self.env.gym.create_camera_sensor(self.env.envs[0], cam_props)
            # initial placement
            look_at = np.array(self.env.root_states[0, :3].cpu(), dtype=np.float64)
            cam_pos = look_at + np.array([1.5, 0.0, 1.0])
            self.env.gym.set_camera_location(self._camera_handle, self.env.envs[0], gymapi.Vec3(*cam_pos), gymapi.Vec3(*look_at))
            self._frame_idx = 0

    def _act(self, obs: torch.Tensor) -> torch.Tensor:
        if self.policy is not None:
            with torch.no_grad():
                return self.policy.act_inference(obs)
        return torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)

    def _update_camera(self):
        if not self._use_viewer:
            return
        look_at = np.array(self.env.root_states[self._track_index, :3].cpu(), dtype=np.float64)
        self._camera_rot = (self._camera_rot + self._camera_rot_per_sec * self.env.dt) % (2 * np.pi)
        rel = 2 * np.array([
            np.cos(self._camera_rot) * self._camera_h_scale,
            np.sin(self._camera_rot) * self._camera_h_scale,
            0.5 * self._camera_v_scale,
        ])
        self.env.set_camera(look_at + rel, look_at, self._track_index)

    def _record_frame(self):
        if self._camera_handle is None:
            return
        # update sensor camera to follow the robot
        look_at = np.array(self.env.root_states[0, :3].cpu(), dtype=np.float64)
        cam_pos = look_at + np.array([1.8, 0.0, 1.1])
        self.env.gym.set_camera_location(self._camera_handle, self.env.envs[0], gymapi.Vec3(*cam_pos), gymapi.Vec3(*look_at))
        # render and write frame every N steps
        if (self._frame_idx % self._frame_stride) == 0:
            self.env.gym.render_all_camera_sensors(self.env.sim)
            out_path = os.path.join(self._video_dir, f"frame_{self._frame_idx:06d}.png")
            self.env.gym.write_camera_image_to_file(self.env.sim, self.env.envs[0], self._camera_handle, gymapi.IMAGE_COLOR, out_path)
        self._frame_idx += 1

    def rollout(self, num_episodes_per_env: int = 2, print_prefix: str = "") -> None:
        # track per-env episode order and reward
        episode_order = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        episode_reward = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        finished_counts = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        total_needed = num_episodes_per_env * self.env.num_envs
        done_episodes = 0

        obs = self.env.get_observations()
        critic_obs = self.env.get_privileged_observations()

        while done_episodes < total_needed:
            actions = self._act(obs)
            step_result = self.env.step(actions)
            if len(step_result) == 4:
                obs, rewards, dones, infos = step_result
                critic_obs = self.env.obs_buf
            else:
                obs, critic_obs, rewards, dones, infos = step_result

            # update viewer camera
            self._update_camera()
            # optionally record frames via sensor camera
            self._record_frame()

            episode_reward += rewards.squeeze(-1) if rewards.dim() > 1 else rewards

            if dones.any():
                env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                for eid in env_ids.tolist():
                    order = int(episode_order[eid].item())
                    rew = float(episode_reward[eid].item())
                    print(f"{print_prefix}env={eid}, ep={order}, reward={rew}")
                    episode_order[eid] += 1
                    finished_counts[eid] += 1
                    done_episodes += 1
                    episode_reward[eid] = 0.0

                # let env handle resets and curriculum
                self.env.training_curriculum()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run rollouts without learning and print per-env rewards")
    parser.add_argument("--task", type=str, default="h1int")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes_per_env", type=int, default=2)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_dir", type=str, default="videos/on_policy_test")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--frame_stride", type=int, default=10)
    args = parser.parse_args()

    runner = OnPolicyTestRunner(task=args.task, num_envs=args.num_envs, headless=args.headless, load_checkpoint=args.load_checkpoint,
                                record_video=args.record_video, video_dir=args.video_dir, camera_width=args.camera_width,
                                camera_height=args.camera_height, frame_stride=args.frame_stride)
    runner.rollout(num_episodes_per_env=args.episodes_per_env)


if __name__ == "__main__":
    main()


