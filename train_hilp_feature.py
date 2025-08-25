#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import typing as tp
import math
import torch.nn.functional as F

try:
    import wandb
except Exception:
    wandb = None

from hilbert_dataset import HilbertRepresentationDataset

def weight_init(m) -> None:
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y

def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")

def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)

class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net: nn.Module = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        return None

def soft_update_params(net, target_net, tau) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

class HILP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, cfg) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.z_dim = z_dim
        self.cfg = cfg

        if self.cfg.feature_type != 'concat':
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", feature_dim]

        self.phi1 = mlp(*layers)
        self.phi2 = mlp(*layers)
        self.target_phi1 = mlp(*layers)
        self.target_phi2 = mlp(*layers)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        self.apply(weight_init)

        # Define a running mean and std
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_std', torch.ones(feature_dim))

    def feature_net(self, obs):
        phi = self.phi1(obs)
        phi = phi - self.running_mean
        return phi

    def value(self, obs: torch.Tensor, goals: torch.Tensor, is_target: bool = False):
        if is_target:
            phi1 = self.target_phi1
            phi2 = self.target_phi2
        else:
            phi1 = self.phi1
            phi2 = self.phi2

        phi1_s = phi1(obs)
        phi1_g = phi1(goals)

        phi2_s = phi2(obs)
        phi2_g = phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        goals = future_obs
        rewards = (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float()
        masks = 1.0 - rewards
        rewards = rewards - 1.0

        next_v1, next_v2 = self.value(next_obs, goals, is_target=True)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.cfg.hilp_discount * masks * next_v

        v1_t, v2_t = self.value(obs, goals, is_target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.cfg.hilp_discount * masks * next_v1
        q2 = rewards + self.cfg.hilp_discount * masks * next_v2
        v1, v2 = self.value(obs, goals, is_target=False)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.cfg.hilp_expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.cfg.hilp_expectile).mean()
        value_loss = value_loss1 + value_loss2

        soft_update_params(self.phi1, self.target_phi1, 0.005)
        soft_update_params(self.phi2, self.target_phi2, 0.005)

        with torch.no_grad():
            phi1 = self.phi1(obs)
            self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
            self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            'hilp/value_loss': value_loss,
            'hilp/v_mean': v.mean(),
            'hilp/v_max': v.max(),
            'hilp/v_min': v.min(),
            'hilp/abs_adv_mean': torch.abs(adv).mean(),
            'hilp/adv_mean': adv.mean(),
            'hilp/adv_max': adv.max(),
            'hilp/adv_min': adv.min(),
            'hilp/accept_prob': (adv >= 0).float().mean(),
        }




def main():
    parser = argparse.ArgumentParser(description="Train ZSRL-style HILP feature on our dataset")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='hilp_feature_runs')
    parser.add_argument('--types', type=str, nargs='*', default=None)
    parser.add_argument('--max_episodes_per_type', type=int, default=2)
    parser.add_argument('--obs_horizon', type=int, default=5)
    parser.add_argument('--goal_future', type=float, default=0.98)
    parser.add_argument('--p_trajgoal', type=float, default=0.5)
    parser.add_argument('--p_randomgoal', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--expectile', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='auto')

    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='hilbert_feature')
    parser.add_argument('--wandb_group', type=str, default='zsrl_hilp')
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device

    # dataset
    dataset = HilbertRepresentationDataset(
        data_dir=args.data_dir,
        types=args.types,
        max_episodes_per_type=args.max_episodes_per_type,
        goal_future=args.goal_future,
        p_trajgoal=args.p_trajgoal,
        p_randomgoal=args.p_randomgoal,
        obs_horizon=args.obs_horizon
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    obs_dim = dataset.obs_dim
    z_dim = 32

    # zsrl hilp
    hilp = HILP(obs_dim=obs_dim, action_dim=1, z_dim=z_dim, hidden_dim=512, cfg=type('cfg', (), {
        'hilp_discount': args.gamma,
        'hilp_expectile': args.expectile,
        'feature_type': 'state',
    }))
    hilp = hilp.to(device)

    optimizer = optim.Adam(hilp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # wandb
    if args.wandb and wandb is not None:
        run_name = args.wandb_run_name or f"hilp_feature_obs{obs_dim}_z{z_dim}"
        wandb.init(project=args.wandb_project, group=args.wandb_group, name=run_name, mode=args.wandb_mode,
                   config={
                       'obs_dim': obs_dim,
                       'z_dim': z_dim,
                       'lr': args.learning_rate,
                       'wd': args.weight_decay,
                       'gamma': args.gamma,
                       'expectile': args.expectile,
                       'obs_horizon': args.obs_horizon,
                       'goal_future': args.goal_future,
                       'p_trajgoal': args.p_trajgoal,
                   })

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        hilp.train()
        epoch_loss = 0.0
        for batch in train_loader:
            obs = batch['obs'].to(device)
            next_obs = batch['next_obs'].to(device)
            goal_obs = batch['goal_obs'].to(device)

            optimizer.zero_grad(set_to_none=True)
            value_loss, info = hilp(obs=obs, action=torch.zeros(obs.size(0), 1, device=device), next_obs=next_obs, future_obs=goal_obs)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(hilp.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(value_loss.item())
            if args.wandb and wandb is not None:
                payload: Dict[str, float] = {k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in info.items()}
                payload['train/iter_loss'] = float(value_loss.item())
                payload['lr'] = optimizer.param_groups[0]['lr']
                wandb.log(payload, step=global_step)
            global_step += 1

        epoch_loss /= max(1, len(train_loader))
        print(f"Epoch {epoch} | loss={epoch_loss:.6f}")

        # save ckpt
        ckpt = {
            'epoch': epoch,
            'hilp_state_dict': hilp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth'))

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()


