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
from tqdm import tqdm

try:    
    import wandb
except Exception:
    wandb = None

from url_benchmark.hilbert_dataset import HilbertRepresentationDataset
from url_benchmark.dataset_zsrl import load_replay_buffer_from_checkpoint, configure_replay_buffer
from url_benchmark.replay_loader_builder import build_replay_loader

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

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor, do_update: bool = True):
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

        if do_update:
            soft_update_params(self.phi1, self.target_phi1, 0.005)
            soft_update_params(self.phi2, self.target_phi2, 0.005)

            with torch.no_grad():
                phi1 = self.phi1(obs)
                self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
                self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            'hilp/value_loss': value_loss.item(),
            'hilp/v_mean': v.mean().item(),
            'hilp/v_max': v.max().item(),
            'hilp/v_min': v.min().item(),
            'hilp/abs_adv_mean': torch.abs(adv).mean().item(),
            'hilp/adv_mean': adv.mean().item(),
            'hilp/adv_max': adv.max().item(),
            'hilp/adv_min': adv.min().item(),
            'hilp/accept_prob': (adv >= 0).float().mean().item(),
        }




def main():
    parser = argparse.ArgumentParser(description="Train ZSRL-style HILP feature on our dataset or ZSRL ReplayBuffer")
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--load_replay_buffer', type=str, default=None)
    parser.add_argument('--discount', type=float, default=0.98)
    parser.add_argument('--future', type=float, default=0.98)
    parser.add_argument('--output_dir', type=str, default='/root/workspace/HILP/hilp_zsrl/runs')
    parser.add_argument('--types', type=str, nargs='*', default=None)
    parser.add_argument('--max_episodes_per_type', type=int, default=None)
    parser.add_argument('--obs_horizon', type=int, default=5)
    parser.add_argument('--goal_future', type=float, default=0.98)
    parser.add_argument('--p_randomgoal', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--expectile', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='auto')

    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='hilp_zsrl')
    parser.add_argument('--wandb_group', type=str, default='zsrl_hilp')
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--z_dim', type=int, default=50)

    args = parser.parse_args()

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device

    # dataset: choose between local Hilbert dataset and ZSRL ReplayBuffer
    use_replay = args.load_replay_buffer is not None
    if use_replay:
        rb = build_replay_loader(
            load_replay_buffer=args.load_replay_buffer,
            replay_buffer_episodes=5000,
            obs_type='state',
            frame_stack=None,
            discount=args.discount,
            future=args.future,
            p_randomgoal=args.p_randomgoal,
            p_currgoal=0.0,
        )
        sample = rb.sample(1)
        obs_dim = int(sample.obs.shape[-1])
    else:
        dataset = HilbertRepresentationDataset(
            data_dir=args.data_dir,
            types=args.types,
            max_episodes_per_type=args.max_episodes_per_type,
            goal_future=args.goal_future,
            p_randomgoal=args.p_randomgoal,
            obs_horizon=args.obs_horizon
        )
        # split train/val similar to train_hilbert_representation.py
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        obs_dim = dataset.obs_dim
    z_dim = args.z_dim

    # build run name from actual parameters and set final output dir = parent/runname
    def _fmt_float(x: float) -> str:
        if x == 0:
            return '0'
        if abs(x) < 1e-3 or abs(x) >= 1e3:
            s = f"{x:.0e}".replace('+', '')
        else:
            s = f"{x:g}"
        return s

    def _replay_tag(path: str) -> str:
        try:
            parts = os.path.normpath(path).split(os.sep)
            if len(parts) >= 3:
                return f"{parts[-3]}_{parts[-2]}"
        except Exception:
            pass
        return os.path.splitext(os.path.basename(path))[0]

    if use_replay:
        data_tag = _replay_tag(args.load_replay_buffer)
        mode_tag = 'rb'
    else:
        data_tag = os.path.basename(os.path.normpath(args.data_dir)) if args.data_dir else 'dataset'
        mode_tag = 'ds'

    runname_parts = [
        'hilp', mode_tag, data_tag,
        f"z{z_dim}", f"bs{args.batch_size}",
        f"lr{_fmt_float(args.learning_rate)}",
        f"g{_fmt_float(args.gamma)}", f"disc{_fmt_float(args.discount)}", f"fut{_fmt_float(args.future)}",
        f"prg{_fmt_float(args.p_randomgoal)}"
    ]
    if args.types:
        runname_parts.append('types-' + '-'.join(args.types))
    if args.obs_horizon:
        runname_parts.append(f"oh{args.obs_horizon}")
    if args.goal_future:
        runname_parts.append(f"gf{_fmt_float(args.goal_future)}")
    runname = '_'.join(runname_parts)

    output_dir = os.path.join(args.output_dir, runname)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # zsrl hilp
    hilp = HILP(obs_dim=obs_dim, action_dim=1, z_dim=z_dim, hidden_dim=512, cfg=type('cfg', (), {
        'hilp_discount': args.gamma,
        'hilp_expectile': args.expectile,
        'feature_type': 'state',
    }))
    hilp = hilp.to(device)

    optimizer = optim.Adam(hilp.parameters(), lr=args.learning_rate)

    # wandb
    if args.wandb and wandb is not None:
        run_name = args.wandb_run_name or runname
        wandb.init(project=args.wandb_project, group=args.wandb_group, name=run_name, mode=args.wandb_mode,
                   config={
                       'obs_dim': obs_dim,
                       'z_dim': z_dim,
                       'lr': args.learning_rate,
                       'gamma': args.gamma,
                       'expectile': args.expectile,
                       'obs_horizon': args.obs_horizon,
                       'goal_future': args.goal_future,
                       'p_randomgoal': args.p_randomgoal,
                       'discount': args.discount,
                       'future': args.future,
                       'types': args.types,
                       'max_episodes_per_type': args.max_episodes_per_type,
                       "data_dir": args.data_dir,
                       "load_replay_buffer": args.load_replay_buffer,
                       "output_dir": args.output_dir,
                       'epochs': args.epochs,
                       'batch_size': args.batch_size,
                   })

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        hilp.train()
        epoch_loss = 0.0
        if use_replay:
            for _ in tqdm(range(1000), desc="Training (ReplayBuffer)"):
                batch = rb.sample(args.batch_size)
                obs = torch.as_tensor(batch.obs, device=device)
                next_obs = torch.as_tensor(batch.next_obs, device=device)
                goal_obs = torch.as_tensor(batch.future_obs, device=device)

                optimizer.zero_grad(set_to_none=True)
                value_loss, info = hilp(obs=obs, action=torch.zeros(obs.size(0), 1, device=device), next_obs=next_obs, future_obs=goal_obs, do_update=True)
                value_loss.backward()
                # torch.nn.utils.clip_grad_norm_(hilp.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(value_loss.item())
                if args.wandb and wandb is not None:
                    payload: Dict[str, float] = {k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in info.items()}
                    payload['train/iter_loss'] = float(value_loss.item())
                    payload['lr'] = optimizer.param_groups[0]['lr']
                    wandb.log(payload, step=global_step)
                global_step += 1

            epoch_loss /= 1000.0
            print(f"Epoch {epoch} | train_loss={epoch_loss:.6f}")

            hilp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in tqdm(range(50), desc="Validation"):
                    vb = rb.sample(args.batch_size)
                    vobs = torch.as_tensor(vb.obs, device=device)
                    vnext = torch.as_tensor(vb.next_obs, device=device)
                    vgoal = torch.as_tensor(vb.future_obs, device=device)
                    vloss, _ = hilp(obs=vobs, action=torch.zeros(vobs.size(0), 1, device=device), next_obs=vnext, future_obs=vgoal, do_update=False)
                    val_loss += float(vloss.item())
            val_loss /= 50.0
            print(f"Epoch {epoch} | val_loss={val_loss:.6f}")
            if args.wandb and wandb is not None:
                wandb.log({'val/loss': val_loss, 'train/epoch_loss': epoch_loss}, step=global_step)
        else:
            for batch in tqdm(train_loader, desc="Training"):
                obs = batch['obs'].to(device)
                next_obs = batch['next_obs'].to(device)
                goal_obs = batch['goal_obs'].to(device)

                optimizer.zero_grad(set_to_none=True)
                value_loss, info = hilp(obs=obs, action=torch.zeros(obs.size(0), 1, device=device), next_obs=next_obs, future_obs=goal_obs, do_update=True)
                value_loss.backward()
                # torch.nn.utils.clip_grad_norm_(hilp.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(value_loss.item())
                if args.wandb and wandb is not None:
                    payload: Dict[str, float] = {k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in info.items()}
                    payload['train/iter_loss'] = float(value_loss.item())
                    payload['lr'] = optimizer.param_groups[0]['lr']
                    wandb.log(payload, step=global_step)
                global_step += 1

            epoch_loss /= max(1, len(train_loader))
            print(f"Epoch {epoch} | train_loss={epoch_loss:.6f}")

            hilp.eval()
            val_loss = 0.0
            val_metrics_accum: Dict[str, float] = {}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    obs = batch['obs'].to(device)
                    next_obs = batch['next_obs'].to(device)
                    goal_obs = batch['goal_obs'].to(device)
                    vloss, vinfo = hilp(obs=obs, action=torch.zeros(obs.size(0), 1, device=device), next_obs=next_obs, future_obs=goal_obs, do_update=False)
                    val_loss += float(vloss.item())
                    for k, v in vinfo.items():
                        val_metrics_accum[k] = val_metrics_accum.get(k, 0.0) + (float(v.item()) if torch.is_tensor(v) else float(v))
            val_loss /= max(1, len(val_loader))
            for k in list(val_metrics_accum.keys()):
                val_metrics_accum[k] /= max(1, len(val_loader))
            print(f"Epoch {epoch} | val_loss={val_loss:.6f}")

            if args.wandb and wandb is not None:
                val_payload: Dict[str, float] = {'val/loss': val_loss}
                for k, v in val_metrics_accum.items():
                    val_payload[f'val/{k}'] = v
                wandb.log(val_payload, step=global_step)

        # save ckpt
        ckpt = {
            'epoch': epoch,
            'hilp_state_dict': hilp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(output_dir, f'ckpt_epoch_{epoch}.pth'))

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()


