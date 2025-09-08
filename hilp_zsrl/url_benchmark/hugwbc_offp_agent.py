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

logger = logging.getLogger(__name__)

# ==============================
# Soft Actor-Critic (SAC) Agent
# ==============================
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    # reuse project factory functions
    from url_benchmark.hugwbc_policy_network import create_hugwbc_policy, create_hugwbc_critic
except Exception as e:
    create_hugwbc_policy = None
    create_hugwbc_critic = None

class SquashedGaussianActor(nn.Module):
    """Simple MLP squashed Gaussian actor.
    Inputs are the concatenation of obs, optional z_vector and clock.
    """
    def __init__(self, in_dim: int, act_dim: int, hidden_dims=(256,256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)  # SAC standard clamp
        return mu, log_std
    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(x)
        std = log_std.exp()
        # reparameterized sample
        noise = torch.randn_like(mu)
        pre_tanh = mu + std * noise
        a = torch.tanh(pre_tanh)
        # log prob with tanh correction
        # log N(pre_tanh; mu, std)
        log_prob = (-0.5 * ((pre_tanh - mu) / (std + 1e-8))**2
                    - log_std
                    - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x.device))).sum(-1, keepdim=True)
        # tanh correction (stable form)
        log_prob -= (2 * (torch.log(torch.tensor(2.0, device=x.device)) - pre_tanh - nn.functional.softplus(-2*pre_tanh))).sum(-1, keepdim=True)
        return a, log_prob
    
    def act_inference(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self(x)
        return torch.tanh(mu)


@dataclass
class HugWBCSACConfig:
    _target_: str = "url_benchmark.hugwbc_offp_agent.HugWBCSACAgent"
    name: str = "hugwbc_sac"
    device: str = "cuda:0"
    # training
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 512
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 0.1
    target_entropy_scale: float = 1.0  # 1.0 => -action_dim
    # architecture
    use_project_actor: bool = False  # True 时使用 create_hugwbc_policy
    z_dim: int = 11
    clock_dim: int = 0
    horizon: int = 5
    # dataset
    data_dir: str = "/path/to/hugwbc/replay/buffer"
    obs_horizon: int = 5
    types: Tuple[str, ...] = ("walk", "turn", "stand")
    max_episodes_per_type: Optional[int] = None
    # logging
    use_wandb: bool = False
    wandb_project: str = "hugwbc_sac"
    wandb_group: str = "experiments"
    wandb_name: str = "hugwbc_sac_training"
    log_dir: str = "./logs/hugwbc_sac"
    num_epochs: int = 200
    eval_every_epochs: int = 5
    save_every_epochs: int = 10


class DoubleQ(nn.Module):
    """Wrap two critics that both take [privileged_obs, action] as input."""
    def __init__(self, critic1: nn.Module, critic2: nn.Module):
        super().__init__()
        self.q1 = critic1
        self.q2 = critic2
    def forward(self, priv: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expect both critics to accept concatenated inputs
        x = torch.cat([priv, act], dim=-1)
        return self.q1(x), self.q2(x)


class HugWBCSACAgent:
    def __init__(self, cfg: HugWBCSACConfig, dataset_cls=None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # dataset（重用你的 SL 数据集包装；batch 必须包含 transitions）
        if dataset_cls is None:
            from url_benchmark.hilbert_dataset import HugWBCSLDataset as DefaultDataset
            dataset_cls = DefaultDataset
        dataset = dataset_cls(
            data_dir=cfg.data_dir,
            obs_horizon=cfg.obs_horizon,
            types=cfg.types,
            max_episodes_per_type=cfg.max_episodes_per_type,
        )
        self.data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # networks 延迟到首个 batch 初始化（拿到维度）
        self.actor = None
        self.critic: Optional[DoubleQ] = None
        self.critic_target: Optional[DoubleQ] = None
        self.actor_opt = None
        self.critic_opt = None
        # temperature
        self.log_alpha = torch.tensor(float(torch.log(torch.tensor(cfg.init_temperature))).item(), device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = None  # 待拿到 act_dim 后设定

    # ---------- helpers ----------
    def _build_networks_from_batch(self, batch: Dict[str, torch.Tensor]):
        # infer dimensions
        obs = batch.get("obs").to(self.device)
        act = batch.get("actions").to(self.device)
        priv = batch.get("privileged_obs").to(self.device)
        z = batch.get("z_vector", torch.zeros(obs.shape[0], 0, device=self.device)).to(self.device)
        clock = batch.get("clock", torch.zeros(obs.shape[0], 0, device=self.device)).to(self.device)
        act_dim = act.shape[-1]
        # actor input uses regular observations + z + clock
        actor_in_dim = obs.shape[-1] + z.shape[-1] + clock.shape[-1]
        # critic uses privileged obs + action (内部拼接)
        critic_in_dim = priv.shape[-1] + act_dim
        # build actor
        if self.cfg.use_project_actor and create_hugwbc_policy is not None:
            self.actor = create_hugwbc_policy(z_dim=z.shape[-1], clock_dim=clock.shape[-1], horizon=self.cfg.horizon).to(self.device)
            self.actor_is_project = True
        else:
            self.actor = SquashedGaussianActor(in_dim=actor_in_dim, act_dim=act_dim).to(self.device)
            self.actor_is_project = False
        # build double critics
        if create_hugwbc_critic is None:
            # fallback: 简单 MLP
            def _mlp(dim_in):
                return nn.Sequential(
                    nn.Linear(dim_in, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, 1)
                )
            q1 = _mlp(critic_in_dim).to(self.device)
            q2 = _mlp(critic_in_dim).to(self.device)
        else:
            # 优先传 input_dim，若工厂无此签名则退化为无参
            try:
                q1 = create_hugwbc_critic(input_dim=critic_in_dim).to(self.device)
                q2 = create_hugwbc_critic(input_dim=critic_in_dim).to(self.device)
            except TypeError:
                q1 = create_hugwbc_critic().to(self.device)
                q2 = create_hugwbc_critic().to(self.device)
        self.critic = DoubleQ(q1, q2).to(self.device)
        # target
        import copy
        self.critic_target = DoubleQ(copy.deepcopy(q1), copy.deepcopy(q2)).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        # optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        # temperature
        self.target_entropy = - self.cfg.target_entropy_scale * act_dim
    
    def _cat_actor_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs"].to(self.device)
        parts = [obs]
        if "z_vector" in batch:
            parts.append(batch["z_vector"].to(self.device))
        if "clock" in batch:
            parts.append(batch["clock"].to(self.device))
        return torch.cat(parts, dim=-1)

    def _actor_sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.actor_is_project:
            # 兼容项目 actor 的 API（若仅有 act_inference，则退化为确定性）
            if hasattr(self.actor, "act"):
                a, logp = self.actor.act(x)  # type: ignore
            elif hasattr(self.actor, "sample"):
                a, logp = self.actor.sample(x)  # type: ignore
            else:
                a = self.actor.act_inference(x) if hasattr(self.actor, "act_inference") else self.actor(x)[0]
                logp = torch.zeros(a.shape[0], 1, device=a.device)
        else:
            a, logp = self.actor.sample(x)
        return a, logp
    
    def _actor_infer(self, x: torch.Tensor) -> torch.Tensor:
        if self.actor_is_project:
            if hasattr(self.actor, "act_inference"):
                return self.actor.act_inference(x)
            elif hasattr(self.actor, "forward"):
                mu, _ = self.actor(x)
                return torch.tanh(mu)
            else:
                raise RuntimeError("Unknown project actor interface.")
        else:
            return self.actor.act_inference(x)
    
    # ---------- SAC updates ----------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.actor is None:
            self._build_networks_from_batch(batch)
        metrics = {}
        device = self.device
        # to device
        for k in list(batch.keys()):
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        obs = batch["obs"]
        next_obs = batch.get("next_obs", None)
        if next_obs is None and "obs_next" in batch:
            next_obs = batch["obs_next"]
        actions = batch["actions"]
        rewards = batch.get("rewards", batch.get("reward")).unsqueeze(-1).float()
        dones = batch.get("dones", batch.get("done")).unsqueeze(-1).float()
        priv = batch["privileged_obs"]
        priv_next = batch.get("next_privileged_obs", None)
        if priv_next is None and "privileged_obs_next" in batch:
            priv_next = batch["privileged_obs_next"]
        # actor inputs
        x = self._cat_actor_input(batch)
        # next actor input
        next_batch = dict(obs=next_obs)
        if "z_vector" in batch: next_batch["z_vector"] = batch["z_vector"]
        if "clock" in batch and "next_clock" in batch: next_batch["clock"] = batch["next_clock"]
        elif "clock" in batch: next_batch["clock"] = batch["clock"]
        x_next = self._cat_actor_input(next_batch)
        # ---------------- Critic ----------------
        with torch.no_grad():
            a_next, logp_next = self._actor_sample(x_next)
            q1_t, q2_t = self.critic_target(priv_next, a_next)
            q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * logp_next
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * q_t
        q1, q2 = self.critic(priv, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        metrics.update({
            "loss/critic": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        })
        # ---------------- Actor & Alpha ----------------
        a, logp = self._actor_sample(x)
        q1_pi, q2_pi = self.critic(priv, a)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp() * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        # temperature
        alpha_loss = (-(self.log_alpha) * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # ---------------- Target update ----------------
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
        metrics.update({
            "loss/actor": actor_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "entropy": (-logp).mean().item(),
        })
        return metrics


def train_sac(cfg: HugWBCSACConfig):
    import time, os, math
    agent = HugWBCSACAgent(cfg)
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, group=cfg.wandb_group, name=cfg.wandb_name, config=dataclasses.asdict(cfg))
    global_step = 0
    for epoch in range(cfg.num_epochs):
        log_sums = {}
        num = 0
        for batch in agent.data_loader:
            metrics = agent.update(batch)
            for k, v in metrics.items():
                log_sums[k] = log_sums.get(k, 0.0) + float(v)
            num += 1
            global_step += 1
        logs = {k: v / max(1, num) for k, v in log_sums.items()}
        logs["epoch"] = epoch
        logs["steps"] = global_step
        if cfg.use_wandb:
            import wandb
            wandb.log(logs, step=global_step)
        else:
            print({k: round(v, 4) for k, v in logs.items()})
        # TODO: 可按 cfg.save_every_epochs 保存 checkpoint
    return agent





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