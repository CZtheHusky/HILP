#!/usr/bin/env python3
"""
希尔伯特表征训练脚本
基于collect_trajectories.py收集的数据格式，训练一个健壮的希尔伯特表征
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from hilbert_dataset import HilbertRepresentationDataset

# optional wandb
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
Note: Dataset class moved to hilbert_dataset.HilbertRepresentationDataset
"""


class HilbertPhiNetwork(nn.Module):
    """
    HILP 表征网络：双 phi 分支（phi1/phi2）。
    提供获取 phi1/phi2 以及基于 phi 的值函数 v(s,g) = -||phi(s)-phi(g)||。
    """
    def __init__(self,
                 obs_dim: int,
                 hidden_dims: List[int],
                 representation_dim: int,
                 use_layer_norm: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.representation_dim = representation_dim

        def make_mlp() -> nn.Sequential:
            layers: List[nn.Module] = []
            input_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, representation_dim))
            return nn.Sequential(*layers)

        self.phi1 = make_mlp()
        self.phi2 = make_mlp()
        # running stats buffers (saved in ckpt via state_dict)
        self.register_buffer('running_mean', torch.zeros(representation_dim))
        self.register_buffer('running_std', torch.ones(representation_dim))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def value(self, obs: torch.Tensor, goals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: [B, D] -> returns (phi1, phi2): [B, Z]"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(-1, self.obs_dim)
        goals = goals.reshape(-1, self.obs_dim)
        phi1_s = self.phi1(obs)
        phi1_g = self.phi1(goals)

        phi2_s = self.phi2(obs)
        phi2_g = self.phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))
        return v1, v2

    def forward(self, obs: torch.Tensor, goals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.value(obs, goals)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.phi1(obs) - self.running_mean


class HilbertRepresentationTrainer:
    """
    希尔伯特表征训练器
    """
    
    def __init__(self, 
                 model: HilbertPhiNetwork,
                 device: str = 'cuda',
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5,
                 wandb_logger=None,
                 use_scheduler: bool = False):
        """
        初始化训练器
        
        Args:
            model: 希尔伯特表征网络
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.wandb = wandb_logger
        self.global_step = 0
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if use_scheduler:
            # 学习率调度器
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=1000,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # 目标网络（软更新）
        self.target_network = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def attach_target(self, target_network: HilbertPhiNetwork):
        self.target_network = target_network.to(self.device)
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()

    @staticmethod
    def expectile_loss(adv: torch.Tensor, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def compute_hilp_loss_steps(self,
                                obs: torch.Tensor,
                                next_obs: torch.Tensor,
                                goal_obs: torch.Tensor,
                                gamma: float,
                                expectile: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """ZSRL 风格：使用单步 (s, next_s, g) 计算 Hilbert expectile TD 损失。
        obs/next_obs/goal_obs: [B, D]
        """
        assert self.target_network is not None, "target network not attached"

        # 计算奖励与 mask（成功=同一点）
        with torch.no_grad():
            success = (torch.norm(obs - goal_obs, dim=-1) < 1e-6).float()
            rewards = success  # 1 if reach goal, else 0
            masks = 1.0 - rewards
            # 与实现一致，偏移到 -1/0 标准（可选）：
            rewards = rewards - 1.0

        # 目标网络 next_v(next_s, g)
        with torch.no_grad():
            next_v1, next_v2 = self.target_network.value(next_obs, goal_obs)
            next_v = torch.minimum(next_v1, next_v2)
            q = rewards + gamma * masks * next_v
            v1_t, v2_t = self.target_network.value(obs, goal_obs)
            v_t = (v1_t + v2_t) / 2
            adv = q - v_t
            q1 = rewards + gamma * masks * next_v1
            q2 = rewards + gamma * masks * next_v2
        v1, v2 = self.model.value(obs, goal_obs)
        v = (v1 + v2) / 2
        # Update model-level running stats from phi1(obs)
        with torch.no_grad():
            z1_s = self.model.encode(obs)
            self.model.running_mean.mul_(0.995).add_(0.005 * z1_s.mean(dim=0).squeeze(0))
            self.model.running_std.mul_(0.995).add_(0.005 * z1_s.std(dim=0, unbiased=False).squeeze(0))
        loss1 = self.expectile_loss(adv, q1 - v1, expectile).mean()
        loss2 = self.expectile_loss(adv, q2 - v2, expectile).mean()
        total_loss = loss1 + loss2

        metrics = {
            'total_loss': float(total_loss.item()),
            'v_mean': float(v.mean().item()),
            'v_max': float(v.max().item()),
            'v_min': float(v.min().item()),
            'abs_adv_mean': float(torch.abs(adv).mean().item()),
            'adv_mean': float(adv.mean().item()),
            'adv_max': float(adv.max().item()),
            'adv_min': float(adv.min().item()),
            'accept_prob': float((adv >= 0).float().mean().item()),
        }
        return total_loss, metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            obs = batch['obs'].to(self.device)
            next_obs = batch['next_obs'].to(self.device)
            goal_obs = batch['goal_obs'].to(self.device)
            # 前向：Hilbert TD 单步损失
            self.optimizer.zero_grad()
            loss, metrics = self.compute_hilp_loss_steps(obs, next_obs, goal_obs, gamma=self.gamma, expectile=self.expectile)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            # 软更新 target
            with torch.no_grad():
                for p, tp in zip(self.model.parameters(), self.target_network.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            
            # 记录损失
            total_loss += loss.item()
            
            # 累积指标
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += value

            # per-batch wandb logging
            if self.wandb is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_payload = {'lr': current_lr}
                for key, value in metrics.items():
                    log_payload[f'train/{key}'] = value
                self.wandb.log(log_payload, step=self.global_step)
            self.global_step += 1
        
        # 计算平均指标
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        return epoch_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                obs = batch['obs'].to(self.device)
                next_obs = batch['next_obs'].to(self.device)
                goal_obs = batch['goal_obs'].to(self.device)
                loss, metrics = self.compute_hilp_loss_steps(obs, next_obs, goal_obs, gamma=self.gamma, expectile=self.expectile)
                total_loss += loss.item()
                
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0
                    val_metrics[key] += value
        
        # 计算平均指标
        num_batches = len(dataloader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def save_checkpoint(self, save_path: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'representation_dim': self.model.representation_dim,
                'obs_dim': self.model.obs_dim
            }
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练希尔伯特表征")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='轨迹数据目录路径')
    parser.add_argument('--output_dir', type=str, default='hilbert_representation_output',
                       help='输出目录')
    parser.add_argument('--representation_dim', type=int, default=32,
                       help='表征维度')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512],
                       help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--trajectory_length', type=int, default=500,
                       help='轨迹长度（时间步）')
    parser.add_argument('--use_layer_norm', action='store_true',
                       help='使用层归一化')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto/cpu/cuda)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    # HILP 超参数
    parser.add_argument('--gamma', type=float, default=0.98, help='Hilbert TD 折扣因子')
    parser.add_argument('--expectile', type=float, default=0.5, help='Hilbert expectile 系数')
    parser.add_argument('--tau', type=float, default=0.005, help='Target 网络软更新系数')
    # goal 采样方式（与 hilp_zsrl/hilp_gcrl 对齐）
    parser.add_argument('--goal_mode', type=str, default='gcrl_mixture', choices=['next','future_geom','gcrl_mixture'],
                       help='目标采样方式：next=下一时刻；future_geom=几何分布抽样未来；gcrl_mixture=同轨迹几何+当前+随机混合')
    parser.add_argument('--goal_future', type=float, default=0.98, help='future_geom 的几何采样参数（类似 zsrl Config.future）')
    parser.add_argument('--p_currgoal', type=float, default=0.0, help='gcrl 混合：当前状态为 goal 的概率')
    parser.add_argument('--p_trajgoal', type=float, default=0.625, help='gcrl 混合：同轨迹未来 goal 的概率')
    parser.add_argument('--p_randomgoal', type=float, default=0.375, help='gcrl 混合：随机 goal 的概率')
    # wandb
    parser.add_argument('--wandb', action='store_true', help='启用 Weights & Biases 日志')
    parser.add_argument('--wandb_project', type=str, default='hilbert_training', help='wandb 项目名')
    parser.add_argument('--wandb_group', type=str, default='debug', help='wandb 分组名')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb 运行名（默认自动）')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'], help='wandb 模式')
    parser.add_argument('--wandb_dir', type=str, default=None, help='wandb 本地目录')
    # 新数据集子集/加速选项
    parser.add_argument('--types', type=str, nargs='*', default=None,
                       help='数据类型列表: constant switch（留空表示自动检测存在的类型）')
    parser.add_argument('--max_episodes_per_type', type=int, default=None,
                       help='每种类型最多加载的episode数（zarr数据集）')
    parser.add_argument('--obs_horizon', type=int, default=5,
                       help='使用的观察历史帧数（1-5）')
    parser.add_argument('--use_scheduler', action='store_true', default=False, help='使用学习率调度器')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    
    # 创建数据集
    logger.info("Creating dataset...")
    dataset = HilbertRepresentationDataset(
        data_dir=args.data_dir,
        types=args.types,
        max_episodes_per_type=args.max_episodes_per_type,
        obs_horizon=args.obs_horizon,
        p_currgoal=args.p_currgoal,
        p_trajgoal=args.p_trajgoal,
        p_randomgoal=args.p_randomgoal,
        goal_future=args.goal_future,
        goal_mode=args.goal_mode
    )
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")    
    # 创建模型
    logger.info("Creating model...")
    detected_obs_dim = int(dataset.obs_dim)
    model = HilbertPhiNetwork(
        obs_dim=detected_obs_dim,
        hidden_dims=args.hidden_dims,
        representation_dim=args.representation_dim,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout
    )
    
    # 创建训练器
    # 初始化 wandb（可选，挪到创建模型后，写入真实 obs_dim）
    if args.wandb:
        run_name = args.wandb_run_name or f"hilbert_obs{detected_obs_dim}_repr{args.representation_dim}_{args.obs_horizon}"
        wandb.init(project=args.wandb_project,
                    group=args.wandb_group,
                    name=run_name,
                    mode=args.wandb_mode,
                    dir=args.wandb_dir,
                    config={
                        'obs_dim': detected_obs_dim,
                        'representation_dim': args.representation_dim,
                        'hidden_dims': args.hidden_dims,
                        'use_layer_norm': args.use_layer_norm,
                        'dropout': args.dropout,
                        'learning_rate': args.learning_rate,
                        'weight_decay': args.weight_decay,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'trajectory_length': args.trajectory_length,
                        'gamma': args.gamma,
                        'expectile': args.expectile,
                        'tau': args.tau,
                        'data_dir': args.data_dir,
                        'types': args.types,
                        'max_episodes_per_type': args.max_episodes_per_type,
                        'obs_horizon': args.obs_horizon,
                        'use_scheduler': args.use_scheduler,
                    })

    trainer = HilbertRepresentationTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        wandb_logger=(wandb if args.wandb else None),
        use_scheduler=args.use_scheduler
    )
    # 绑定 HILP 目标网络与超参数
    target_model = HilbertPhiNetwork(
        obs_dim=detected_obs_dim,
        hidden_dims=args.hidden_dims,
        representation_dim=args.representation_dim,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout
    )
    trainer.attach_target(target_model)
    trainer.gamma = args.gamma
    trainer.expectile = args.expectile
    trainer.tau = args.tau
    trainer.goal_mode = args.goal_mode
    trainer.goal_future = args.goal_future
    trainer.p_currgoal = args.p_currgoal
    trainer.p_trajgoal = args.p_trajgoal
    trainer.p_randomgoal = args.p_randomgoal
    
    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # 训练循环
    logger.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader)
        
        # 验证
        val_metrics = trainer.validate(val_loader)
        
        # 记录损失
        trainer.train_losses.append(train_metrics['total_loss'])
        trainer.val_losses.append(val_metrics['total_loss'])
        
        # 打印指标
        logger.info(f"Train Loss: {train_metrics['total_loss']:.6f}")
        logger.info(f"Val Loss: {val_metrics['total_loss']:.6f}")
        
        # 保存最佳模型
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            trainer.save_checkpoint(best_model_path, epoch)
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            trainer.save_checkpoint(checkpoint_path, epoch)
        
        # 保存训练历史
        history = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'epochs': list(range(1, len(trainer.train_losses) + 1))
        }
        
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    trainer.save_checkpoint(final_model_path, args.epochs)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(trainer.train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(trainer.val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.wandb and wandb is not None:
        wandb.log({'best/val_total_loss': best_val_loss})
        wandb.finish()


if __name__ == "__main__":
    main()
