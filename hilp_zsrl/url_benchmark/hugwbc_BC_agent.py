#!/usr/bin/env python3
"""
HugWBC BC Agent - Behavioral Cloning Agent for HugWBC Policy Network
====================================================================

This module implements a Behavioral Cloning (BC) agent that learns to imitate
expert demonstrations from HugWBC replay buffer data using the HugWBCPolicyNetwork.

Author: Created for HugWBC BC training
License: BSD-3-Clause
"""
from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
from url_benchmark.legged_gym_env_utils import _make_eval_env, _to_rgb_frame
import os
import logging
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import omegaconf
from hydra.core.config_store import ConfigStore
import wandb
from tqdm import tqdm

from url_benchmark import utils
from url_benchmark.hilbert_dataset import HugWBCSLDataset
from url_benchmark.hugwbc_policy_network import create_hugwbc_policy, create_hugwbc_critic
from isaacgym import gymapi
import imageio.v2 as imageio
from collections import defaultdict
import torch.nn.functional as F
import argparse
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def timer(name="Block"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] 耗时: {end - start:.4f} 秒")


CANDATE_ENV_COMMANDS = {
    "default": [0, 0, 0, 2, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_backward_walk": [-0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_forward_walk": [0.6, 0, 0, 1, 0.15, 0.5, 0.2, 0, 0, 0],
    "slow_backward_walk_low_height": [-0.6, 0, 0, 1, 0.15, 0.5,-0.3, 0, 0, 0],
    "slow_forward_walk_low_height": [0.6, 0, 0, 1, 0.15, 0.5, -0.3, 0, 0, 0],
    "fast_walk": [2, 0, 0, 2.5, 0.15, 0.5, 0.3, 0, 0, 0],
    "slow_turn": [0.5, 0, -0.5, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
    "fast_turn": [0.5, 0, 0.5, 2.5, 0.15, 0.5, 0.2, 0, 0, 0],
    "crab_right_walk": [0, 0.6, 0, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
    "crab_left_walk": [0, -0.6, 0, 1.5, 0.15, 0.5, 0.2, 0, 0, 0],
}

@dataclasses.dataclass
class HugWBCBCAgentConfig:
    """Configuration for HugWBC BC Agent."""
    _target_: str = "url_benchmark.hugwbc_BC_agent.HugWBCBCAgent"
    name: str = "hugwbc_bc"
    
    # Network architecture
    z_dim: int = 11
    horizon: int = 5
    clock_dim: int = 0
    
    # Training parameters
    lr: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 100
    log_every_steps: int = 100
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    
    # Data parameters
    data_dir: str = "/path/to/hugwbc/replay/buffer"
    obs_horizon: int = 5
    max_episodes_per_type: Optional[int] = None
    types: Optional[List[str]] = None
    use_mse: bool = False
    # Device
    device: str = "cuda"
    use_neg_logp: bool = False
    pretrained: bool = False

    
    # Logging
    use_wandb: bool = True
    log_dir: str = "./logs/hugwbc_bc"
    wandb_project: str = "hugwbc_bc"
    wandb_group: str = "experiments"
    wandb_name: str = "hugwbc_bc_training"


cs = ConfigStore.instance()
cs.store(group="agent", name="hugwbc_bc", node=HugWBCBCAgentConfig)


class HugWBCBCAgent:
    """Behavioral Cloning Agent for HugWBC Policy Network."""
    
    def __init__(self, cfg: HugWBCBCAgentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Create policy network
        self.policy = create_hugwbc_policy(
            z_dim=cfg.z_dim, 
            clock_dim=cfg.clock_dim,
            horizon=cfg.horizon
        ).to(self.device)
        if cfg.pretrained:
            state_dict = torch.load("/root/workspace/HugWBC/logs/h1_interrupt/Aug21_13-31-13_/model_40000.pt")['model_state_dict']
            raw_state_dict = self.policy.state_dict()
            for k in raw_state_dict.keys():
                if k in state_dict.keys() and raw_state_dict[k].shape == state_dict[k].shape:
                    raw_state_dict[k] = state_dict[k]
                else:
                    print(f"Missing state dict key: {k}, using initialized value.")
            self.policy.load_state_dict(raw_state_dict)
        # Create optimizer and scheduler
        self.optimizer = Adam(self.policy.parameters(), lr=cfg.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.5)
        with timer("Loading dataset"):
            dataset = HugWBCSLDataset(
                data_dir=cfg.data_dir,
                obs_horizon=cfg.obs_horizon,
                types=cfg.types,
                max_episodes_per_type=cfg.max_episodes_per_type,
            )
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
        self.data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        self.val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize wandb
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                group=self.cfg.wandb_group,
                name=self.cfg.wandb_name,
            )
        
        logger.info(f"Initialized HugWBC BC Agent with {sum(p.numel() for p in self.policy.parameters())} parameters")
        device_id = int(cfg.device.split(":")[-1]) if "cuda:" in cfg.device else 0
        env = _make_eval_env(device_id)
        W, H, FPS = 720, 720, 30
        cam_props = gymapi.CameraProperties()
        cam_props.width = W
        cam_props.height = H
        self.W = W
        self.H = H
        self.FPS = FPS
        self.cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
        camera_relative_position = np.array([1, 0, 0.8])
        for i in range(env.num_bodies):
            env.gym.set_rigid_body_color(
                env.envs[0], env.actor_handles[0], i,
                gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3)
            )
        track_index = 0
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        env.set_camera(look_at + camera_relative_position, look_at, track_index)
        self.eval_env = env
        self.camera_relative_position = camera_relative_position
        self.track_index = track_index
        self.look_at = look_at
        self.camera_rot = np.pi * 8 / 10
        self.camera_rot_per_sec = 1 * np.pi / 10
        self.obs_dim = 63


    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, 
                    z_vector: torch.Tensor, privileged_obs: torch.Tensor, clock: torch.Tensor=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute BC loss.
        
        Args:
            obs: Observations [B, horizon, feature_dim]
            actions: Expert actions [B, action_dim]
            z_vector: Goal vector [B, z_dim]
            privileged_obs: Privileged observations [B, 3]
            clock: Clock observations [B, 2]
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Forward pass through policy
        self.policy.update_distribution(obs, z_vector, privileged_obs, clock=clock, sync_update=True)
        dist = self.policy.distribution
        mean_action = dist.mean
        
        # Compute action log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1)  # [B]
        
        # BC loss: maximize log probability of expert actions
        neg_log_p = -log_probs.mean()
        mse_loss = F.mse_loss(mean_action, actions)
        
        # Get privileged reconstruction loss
        privileged_recon_loss = self.policy.actor.privileged_recon_loss
        
        # Total loss
        total_loss = privileged_recon_loss
        if self.cfg.use_neg_logp:
            total_loss = total_loss + neg_log_p
        if self.cfg.use_mse:
            total_loss = total_loss + mse_loss
        
        # Metrics
        metrics = {
            'privileged_recon_loss': privileged_recon_loss.item(),
            'total_loss': total_loss.item(),
            'log_prob_mean': log_probs.mean().item(),
            'log_prob_std': log_probs.std().item(),
            'mse_loss': mse_loss.item(),
        }
        
        return total_loss, metrics

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Batch from data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.policy.train()

        obs, actions, z_vector, privileged_obs, clock = batch['obs'].cuda(), batch['actions'].cuda(), batch['z_vector'].cuda(), batch['privileged_obs'].cuda(), batch['clock'].cuda()
        
        # Compute loss
        loss, metrics = self.compute_loss(obs, actions, z_vector, privileged_obs, clock=clock)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update step counter
        self.global_step += 1
        
        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the policy on a validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        
        val_metrics = defaultdict(list)
        for val_batch in tqdm(self.val_data_loader, desc="Evaluating", total=len(self.val_data_loader)):
            # Prepare batch
            obs, actions, z_vector, privileged_obs, clock = val_batch['obs'].cuda(), val_batch['actions'].cuda(), val_batch['z_vector'].cuda(), val_batch['privileged_obs'].cuda(), val_batch['clock'].cuda()
            
            # Compute loss
            loss, metrics = self.compute_loss(obs, actions, z_vector, privileged_obs, clock=clock)
            
            # Add validation prefix
            for k, v in metrics.items():
                val_metrics[k].append(v)
        ret_metrics = {}
        for k, v in val_metrics.items():
            ret_metrics[f'val/{k}'] = np.mean(v)
                        
        return ret_metrics

    def env_eval(self):
        eval_time = 10
        video_save_parent = f"{self.cfg.log_dir}/{self.global_step}"
        os.makedirs(video_save_parent, exist_ok=True)
        eval_steps = int(eval_time / self.eval_env.dt)
        ret_metrics = {}
        eval_results = []
        eval_ep_rewards = []
        for key, command_vec in CANDATE_ENV_COMMANDS.items():
            command_name = key
            command_vec = torch.as_tensor(command_vec, device=self.eval_env.device)
            VIDEO_PATH = f"{video_save_parent}/{command_name}.mp4"
            _, _ = self.eval_env.reset()
            self.eval_env.commands[:, :10] = command_vec
            obs, critic_obs, _, _, _ = self.eval_env.step(torch.zeros(
                self.eval_env.num_envs, self.eval_env.num_actions, dtype=torch.float, device=self.eval_env.device))
            look_at = np.array(self.eval_env.root_states[0, :3].cpu(), dtype=np.float64)
            self.eval_env.set_camera(look_at + self.camera_relative_position, look_at, self.track_index)
            # ==================== 相机传感器与视频写出器 ====================
            frame_skip = max(1, int(round(1.0 / (self.FPS * self.eval_env.dt))))  # 模拟 -> 视频帧率下采样


            # 让传感器相机与 viewer 相机初始对齐
            self.eval_env.gym.set_camera_location(
                self.cam_handle, self.eval_env.envs[0],
                gymapi.Vec3(*(look_at + self.camera_relative_position)),
                gymapi.Vec3(*look_at)
            )
            timestep = 0
            proprior_obs = obs[..., :self.obs_dim]
            obs_cmd = obs[:, -1, self.obs_dim:self.obs_dim + 11]
            clock = obs[..., -1, self.obs_dim + 11:self.obs_dim + 13]

            # imageio 写 mp4，依赖 ffmpeg（imageio-ffmpeg）
            writer = imageio.get_writer(VIDEO_PATH, fps=self.FPS)
            dones = np.zeros(self.eval_env.num_envs, dtype=np.bool_)
            ep_rewards = 0
            ep_rewards_list = []
            while not dones.any() and timestep < eval_steps:
                with torch.inference_mode():
                    actions, _ = self.policy.act_inference(proprior_obs, z_vector=obs_cmd, clock=clock, privileged_obs=critic_obs)
                obs, critic_obs, reward, dones, _ = self.eval_env.step(actions)
                
                proprior_obs = obs[..., :self.obs_dim]
                obs_cmd = obs[:, -1, self.obs_dim:self.obs_dim + 11]
                clock = obs[..., -1, self.obs_dim + 11:self.obs_dim + 13]
                self.eval_env.commands[:, :10] = command_vec

                ep_rewards_list.append(float(reward.item()))
                ep_rewards += reward.item()
                # print(f"z_reward: {z_reward}, env_reward: {reward.item()}")
                if dones.any():
                    print(f"command: {command_name}, episode env reward: {ep_rewards}, mean_step_reward: {np.mean(ep_rewards_list)}")
                # ===== 相机跟踪与旋转 =====
                look_at = np.array(self.eval_env.root_states[0, :3].cpu(), dtype=np.float64)
                camera_rot = (self.camera_rot + self.camera_rot_per_sec * self.eval_env.dt) % (2 * np.pi)
                h_scale, v_scale = 1.0, 0.8
                camera_relative_position = 2 * np.array(
                    [np.cos(camera_rot) * h_scale, np.sin(camera_rot) * h_scale, 0.5 * v_scale]
                )
                # 更新 viewer 相机
                self.eval_env.set_camera(look_at + camera_relative_position, look_at, self.track_index)
                # env.set_camera(look_at + camera_relative_position, look_at, track_index)
                # 同步传感器相机（用于录制）
                self.eval_env.gym.set_camera_location(
                    self.cam_handle, self.eval_env.envs[0],
                    gymapi.Vec3(*(look_at + camera_relative_position)),
                    gymapi.Vec3(*look_at)
                )

                # ===== 干扰和中断设置 =====
                self.eval_env.use_disturb = True
                self.eval_env.disturb_masks[:] = True
                self.eval_env.disturb_isnoise[:] = True
                self.eval_env.disturb_rad_curriculum[:] = 1.0
                self.eval_env.interrupt_mask[:] = self.eval_env.disturb_masks[:]
                self.eval_env.standing_envs_mask[:] = True

                # ===== 抓帧与写视频（CPU 路径）=====
                if timestep % frame_skip == 0:
                    # 确保渲染管线推进
                    self.eval_env.gym.step_graphics(self.eval_env.sim)
                    self.eval_env.gym.render_all_camera_sensors(self.eval_env.sim)

                    img_any = self.eval_env.gym.get_camera_image(
                        self.eval_env.sim, self.eval_env.envs[0], self.cam_handle, gymapi.IMAGE_COLOR
                    )
                    rgb = _to_rgb_frame(img_any, self.H, self.W)
                    writer.append_data(rgb)
                timestep += 1
            writer.close()
            print("Saved video to %s", VIDEO_PATH)
            mean_env_rewards = np.mean(ep_rewards_list)
            ret_metrics[f"env_eval_detail/{command_name}_per_step"] = mean_env_rewards
            ret_metrics[f"env_eval_detail/{command_name}_episode"] = ep_rewards
            eval_results.append(mean_env_rewards)
            eval_ep_rewards.append(ep_rewards)
        ret_metrics[f"env_eval/rewards"] = np.mean(eval_results)
        ret_metrics[f"env_eval/episode_rewards"] = np.mean(eval_ep_rewards)
        return ret_metrics


    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting HugWBC BC training...")
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        print("Data loader speed test")
        num_batches = len(self.data_loader)
        for batch in tqdm(self.data_loader, desc="Training", total=num_batches):
            pass
        print("Data loader speed test finished")

        
        # Training loop
        for epoch in range(self.cfg.num_epochs):
            self.current_epoch = epoch
            # Evaluation
            if (epoch) % self.cfg.eval_every_epochs == 0:
                env_eval_metrics = self.env_eval()
                val_metrics = self.evaluate()

                
                # Log validation metrics to wandb
                if self.cfg.use_wandb:
                    wandb.log(env_eval_metrics, step=self.global_step)
                    wandb.log(val_metrics, step=self.global_step)
                
                
                logger.info(f"Epoch {epoch} Evaluation - "
                          f"Val Total Loss: {val_metrics['val/total_loss']:.6f}, "
                          f"Env Eval Rewards: {env_eval_metrics['env_eval/rewards']:.6f}, "
                          f"Env Eval Episode Rewards: {env_eval_metrics['env_eval/episode_rewards']:.6f}")
            # Training phase
            epoch_metrics = []
            num_batches = len(self.data_loader)
            
            for batch in tqdm(self.data_loader, desc="Training", total=num_batches):
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Log training metrics
                if self.global_step % self.cfg.log_every_steps == 0:
                    # Average metrics over recent steps
                    recent_metrics = epoch_metrics[-self.cfg.log_every_steps:]
                    avg_metrics = {}
                    for key in recent_metrics[0].keys():
                        avg_metrics["train/" + key] = np.mean([m[key] for m in recent_metrics])
                    
                    # Log to wandb
                    if self.cfg.use_wandb:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    # Log to console
                    logger.info(f"Epoch {epoch}, Step {self.global_step}, "
                              f"Total Loss: {avg_metrics['train/total_loss']:.6f}")
                    
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Log epoch-level metrics to wandb
            if self.cfg.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
        
        logger.info("Training completed!")
        
        # Close wandb run
        if self.cfg.use_wandb:
            wandb.finish()

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        os.makedirs(Path(self.cfg.log_dir) / "checkpoints", exist_ok=True)
        checkpoint_path = Path(self.cfg.log_dir) / "checkpoints" / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': self.cfg,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log checkpoint path to wandb
        if self.cfg.use_wandb:
            wandb.save(str(checkpoint_path))

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def act(self, obs: torch.Tensor, z_vector: torch.Tensor, 
            privileged_obs: Optional[torch.Tensor] = None, clock: torch.Tensor = None) -> torch.Tensor:
        """Get action from policy.
        
        Args:
            obs: Observations [B, horizon, feature_dim]
            z_vector: Goal vector [B, z_dim]
            privileged_obs: Privileged observations [B, 3] (optional)
            
        Returns:
            Actions [B, action_dim]
        """
        self.policy.eval()
        
        with torch.no_grad():
            actions, _ = self.policy.act_inference(obs, z_vector, clock=clock)
        
        return actions

# CUDA_VISIBLE_DEVICES=1 python ./url_benchmark/hugwbc_BC_agent.py
def main(args):
    # Create config
    assert args.use_neg_logp or args.use_mse, "Please use either negative log prob loss or mean square error loss."
    cfg = HugWBCBCAgentConfig(
        data_dir=args.data_dir,
        log_dir=f"./logs/{args.log_dir}",
        use_wandb=True,
        wandb_project="hugwbc_bc",
        wandb_group="hugwbc_bc",
        wandb_name=f"{args.log_dir}",
        num_epochs=100,
        batch_size=1024,
        lr=1e-4,
        device=args.device,
        use_neg_logp=args.use_neg_logp,
        use_mse=args.use_mse,
        pretrained=args.pretrained,
        clock_dim=2,
    )
    
    # Create and train agent
    agent = HugWBCBCAgent(cfg)
    agent.train()


parse = argparse.ArgumentParser()
parse.add_argument("--device", type=str, default="cuda", help="Path to config file.")
parse.add_argument("--use_neg_logp", action="store_true", default=False, help="Whether to use negative log prob loss.")
parse.add_argument("--use_mse", action="store_true", default=False, help="Whether to use mean square error loss.")
parse.add_argument("--pretrained", action="store_true", default=False, help="Whether to use pretrained model.")
parse.add_argument("--log_dir", type=str, default="hugwbc_bc", help="Path to logs.")
parse.add_argument("--data_dir", type=str, default="/root/workspace/HugWBC/dataset/collected_trajectories_v2", help="Path to data.")

# python ./url_benchmark/hugwbc_BC_agent.py --device cuda:0 --use_neg_logp --log_dir hugwbc_bc_neg_mse --use_mse
# python ./url_benchmark/hugwbc_BC_agent.py --device cuda:1 --use_neg_logp --pretrained --log_dir hugwbc_bc_neg_pretrain_mse --use_mse
# python ./url_benchmark/hugwbc_BC_agent.py --device cuda:1 --use_neg_logp --pretrained --log_dir hugwbc_bc_neg_pretrain
# python ./url_benchmark/hugwbc_BC_agent.py --device cuda:2 --pretrained --log_dir hugwbc_bc_pretrain_mse --use_mse
# python ./url_benchmark/hugwbc_BC_agent.py --device cuda:3 --log_dir hugwbc_bc_mse
# python ./url_benchmark/hugwbc_BC_agent.py --device cuda:0 --use_neg_logp  --use_mse  --log_dir hugwbc_bc_neg_mse_large --data_dir /root/workspace/HugWBC/dataset/collected_large



if __name__ == "__main__":
    args = parse.parse_args()
    main(args)