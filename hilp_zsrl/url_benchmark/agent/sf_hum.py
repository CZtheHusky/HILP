import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.utils import utils
from url_benchmark.dataset_utils.in_memory_replay_buffer import ReplayBuffer
from .ddpg import MetaDict, make_aug_encoder
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, mlp, OnlineCov
from url_benchmark.dmc_utils.dmc import TimeStep
from url_benchmark.hugwbc_models.hugwbc_policy_network import create_hugwbc_policy
# from url_benchmark.hugwbc_models.hugwbc_policy_sac_network import create_sac_policy
from url_benchmark.hugwbc_models.hugwbc_policy_online_network import create_sac_policy
import time
from typing import Dict, Any, Iterable, Tuple, Optional


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SFHumanoidAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.sf_hum.SFHumanoidAgent"
    name: str = "sf"
    obs_type: str = omegaconf.MISSING  # to be specified later
    image_wh: int = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    critic_obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 5
    sf_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 1
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 10000
    hidden_dim: int = 1024   # 128, 2048
    phi_hidden_dim: int = 512   # 128, 2048
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 50  # 30-200
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"  # 0,  0.1, 0.2
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    nstep: int = 1
    batch_size: int = 1024
    init_sf: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    debug: bool = False
    preprocess: bool = True
    num_sf_updates: int = 1
    feature_learner: str = "hilp"
    mix_ratio: float = 0.5
    q_loss: bool = True
    update_cov_every_step: int = 1000
    add_trunk: bool = False
    command_injection: bool = False # whether to generate command-conditioned z
    use_raw_command: bool = False # whether to use raw command

    feature_type: str = 'state'  # 'state', 'diff', 'concat'
    hilp_discount: float = 0.98
    hilp_expectile: float = 0.5
    train_phi_only: bool = False
    use_sac_net: bool = False
    obs_horizon: int = 5
    random_sample_z: bool = True


cs = ConfigStore.instance()
cs.store(group="agent", name="sf_hum", node=SFHumanoidAgentConfig)


class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net: nn.Module = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        return None


class Identity(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.feature_net = nn.Identity()


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

        layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", hidden_dim, "relu", hidden_dim, "relu", feature_dim]

        self.phi1 = mlp(*layers)
        self.phi2 = mlp(*layers)
        self.target_phi1 = mlp(*layers)
        self.target_phi2 = mlp(*layers)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        self.apply(utils.weight_init)

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
        utils.soft_update_params(self.phi1, self.target_phi1, 0.005)
        utils.soft_update_params(self.phi2, self.target_phi2, 0.005)
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


class Laplacian(FeatureLearner):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        loss = (phi - next_phi).pow(2).mean()
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class ContrastiveFeature(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del next_obs
        assert future_obs is not None
        phi = self.feature_net(obs)
        future_mu = self.mu_net(future_obs)
        phi = F.normalize(phi, dim=1)
        future_mu = F.normalize(future_mu, dim=1)
        logits = torch.einsum('sd, td-> st', phi, future_mu)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = - logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ContrastiveFeaturev2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del next_obs
        assert future_obs is not None
        future_phi = self.feature_net(future_obs)
        mu = self.mu_net(obs)
        future_phi = F.normalize(future_phi, dim=1)
        mu = F.normalize(mu, dim=1)
        logits = torch.einsum('sd, td-> st', mu, future_phi)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = - logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ICM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim, 'tanh')
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        return icm_loss


class TransitionModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_obs).pow(2).mean()
        return forward_error


class TransitionLatentModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        with torch.no_grad():
            next_phi = self.target_feature_net(next_obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_phi.detach()).pow(2).mean()
        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)

        return forward_error


class AutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.decoder = mlp(z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        del next_obs
        del action
        phi = self.feature_net(obs)
        predicted_obs = self.decoder(phi)
        reconstruction_error = (predicted_obs - obs).pow(2).mean()
        return reconstruction_error


class SVDSR(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.target_mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        mu = self.mu_net(next_obs)
        SR = torch.einsum("sd, td -> st", phi, mu)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_phi, target_mu)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = - 2 * SR.diag().mean() + (SR - 0.99 * target_SR.detach())[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utils.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDSRv2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.target_mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(obs)
        SR = torch.einsum("sd, td -> st", mu, phi)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_mu, target_phi)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = - 2 * SR.diag().mean() + (SR - 0.98 * target_SR.detach())[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utils.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim + action_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(torch.cat([obs, action], dim=1))
        P = torch.einsum("sd, td -> st", mu, phi)
        I = torch.eye(*P.size(), device=P.device)
        off_diag = ~I.bool()
        loss = - 2 * P.diag().mean() + P[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class SFHumanoidAgent:

    def __init__(self, **kwargs: tp.Any):
        cfg = SFHumanoidAgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None

        self.command_injection = cfg.command_injection
        self.use_raw_command = cfg.use_raw_command

        # models
        if cfg.obs_type == 'pixels':
            self.aug, self.encoder = make_aug_encoder(cfg)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0]
        self.critic_obs_dim = cfg.critic_obs_shape[0]
        if cfg.feature_learner == "identity":
            cfg.z_dim = self.obs_dim
            self.cfg.z_dim = self.obs_dim
        # create the network
        # if self.cfg.boltzmann:
        #     self.actor: nn.Module = DiagGaussianActor(cfg.obs_type, self.obs_dim, cfg.z_dim, self.action_dim,
        #                                               cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        # else:
        #     self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
        #                        cfg.feature_dim, cfg.hidden_dim,
        #                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if self.cfg.use_sac_net:
            self.actor = create_sac_policy(z_dim=cfg.z_dim, horizon=cfg.obs_horizon, proprio_dim=self.obs_dim // cfg.obs_horizon, clock_dim=0).to(cfg.device)
        else:
            self.actor = create_hugwbc_policy(z_dim=cfg.z_dim, horizon=cfg.obs_horizon, proprio_dim=self.obs_dim // cfg.obs_horizon, clock_dim=0).to(cfg.device)
        self.successor_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim,
                                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # build up the target network
        self.successor_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim,
                                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)

        learner = dict(icm=ICM, transition=TransitionModel, latent=TransitionLatentModel,
                       contrastive=ContrastiveFeature, autoencoder=AutoEncoder, lap=Laplacian,
                       random=FeatureLearner, svd_sr=SVDSR, svd_p=SVDP,
                       contrastivev2=ContrastiveFeaturev2, svd_srv2=SVDSRv2,
                       identity=Identity, hilp=HILP)[self.cfg.feature_learner]
        extra_kwargs = dict()
        if self.cfg.feature_learner == 'hilp':
            extra_kwargs = dict(
                cfg=self.cfg,
            )
        self.feature_learner = learner(self.obs_dim, self.action_dim, cfg.z_dim, cfg.phi_hidden_dim, **extra_kwargs).to(cfg.device)

        print("Successor net: ", self.successor_net)
        print("feature learner: ", self.feature_learner)

        # load the weights into the target networks
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        if self.command_injection:
            self.command_injection_net = mlp(11, cfg.hidden_dim, "ntanh", cfg.hidden_dim, "relu", cfg.z_dim).to(cfg.device)
            self.command_injection_opt = torch.optim.Adam(self.command_injection_net.parameters(), lr=cfg.lr)
            self.command_injection_net.train()
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.feature_learner not in ["random", "identity"]:
            self.phi_opt = torch.optim.Adam(self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr)
        self.train()
        self.successor_target_net.train()

        self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.successor_net]:
            net.train(training)
        if self.phi_opt is not None:
            self.feature_learner.train()

    # def init_from(self, other) -> None:
    #     # copy parameters over
    #     names = ["encoder", "actor"]
    #     if self.cfg.init_sf:
    #         names += ["successor_net", "feature_learner", "successor_target_net"]
    #     for name in names:
    #         utils.hard_update_params(getattr(other, name), getattr(self, name))
    #     for key, val in self.__dict__.items():
    #         if isinstance(val, torch.optim.Optimizer):
    #             val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def init_from(self, other) -> None:
        # 1) 只复制权重/缓冲，不替换模块对象
        device = torch.device(self.cfg.device)
        print("Agent Device:", device)

        # -------- helpers: 自动收集 --------
        def _collect_modules(obj):
            for k, v in obj.__dict__.items():
                if isinstance(v, nn.Module):
                    yield k, v

        def _collect_free_params(obj):
            for k, v in obj.__dict__.items():
                if isinstance(v, torch.nn.Parameter):
                    yield k, v

        def _collect_optimizers(obj):
            for k, v in obj.__dict__.items():
                if isinstance(v, torch.optim.Optimizer):
                    yield k, v

        # -------- 1) 复制所有子模块权重/缓冲（自动发现） --------
        for name, dst_mod in _collect_modules(self):
            src_mod = getattr(other, name, None)
            if isinstance(src_mod, nn.Module):
                try:
                    # strict=False 更健壮，允许 shape/键有少量出入
                    dst_mod.load_state_dict(src_mod.state_dict(), strict=True)
                    # 也可打印一下 missed/unused keys 方便调试：
                    # print(f"[{name}] loaded.")
                except Exception as e:
                    print(f"Error loading module '{name}': {e}")

        # -------- 1.1) 复制“游离”的 nn.Parameter --------
        for name, dst_p in _collect_free_params(self):
            src_p = getattr(other, name, None)
            if isinstance(src_p, torch.nn.Parameter):
                try:
                    with torch.no_grad():
                        dst_p.copy_(src_p.data)
                    # print(f"[param {name}] copied.")
                except Exception as e:
                    print(f"Error copying param '{name}': {e}")

        # -------- 2) 统一迁移所有子模块到 device --------
        for _, m in _collect_modules(self):
            m.to(device)
        # 游离参数也迁移（通常你不会有很多，但以防万一）
        for name, p in _collect_free_params(self):
            if p.device != device:
                with torch.no_grad():
                    p.data = p.data.to(device)

        # -------- 3) 复制优化器状态并把其 state 迁移到 device（自动发现） --------
        for key, opt in _collect_optimizers(self):
            src_opt = getattr(other, key, None)
            if isinstance(src_opt, torch.optim.Optimizer):
                try:
                    opt.load_state_dict(copy.deepcopy(src_opt.state_dict()))
                    # 把优化器 state 张量搬到目标设备
                    for state in opt.state.values():
                        for sk, sv in state.items():
                            if isinstance(sv, torch.Tensor):
                                state[sk] = sv.to(device, non_blocking=True)
                    # print(f"[optimizer {key}] loaded & moved.")
                except Exception as e:
                    print(f"Error loading optimizer '{key}': {e}")

        # -------- 4) 记录设备（若你需要） --------
        self.device = str(device)

        # -------- 5) 可选：一致性自检（调试期很有用） --------
        def _check_opt(opt, name):
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        assert p.device == device, f"{name}: param on {p.device}, expected {device}"
                        if p.grad is not None:
                            assert p.grad.device == device, f"{name}: grad on {p.grad.device}, expected {device}"
            for s in opt.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        assert v.device == device, f"{name}: state[{k}] on {v.device}, expected {device}"

        for key, opt in _collect_optimizers(self):
            _check_opt(opt, key)

    def get_goal_meta(self, goal_array: np.ndarray, obs_array: np.ndarray = None) -> MetaDict:
        assert self.cfg.feature_learner == 'hilp'

        desired_goal = torch.as_tensor(goal_array).to(self.cfg.device)
        if len(desired_goal.shape) == 1:
            desired_goal = desired_goal.unsqueeze(0)
        if obs_array is not None:
            obs = torch.as_tensor(obs_array).to(self.cfg.device)
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            with torch.no_grad():
                obs = self.encoder(obs)
                desired_goal = self.encoder(desired_goal)
                z_g = self.feature_learner.feature_net(desired_goal)
                z_s = self.feature_learner.feature_net(obs)

            z = (z_g - z_s)
        else:
            with torch.no_grad():
                desired_goal = self.encoder(desired_goal)
                z_g = self.feature_learner.feature_net(desired_goal)
                z = z_g
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        # check if there is nan
        if torch.isnan(z).any():
            # find the nan
            nan_idx = torch.isnan(z).any()
            print(f"z is nan at index {nan_idx}")
            # nan z_g, z_s
            nan_z_g = z_g[nan_idx]
            print(f"nan z_g: {nan_z_g}")
            print(f"nan goal: {desired_goal[nan_idx]}")
            try:
                nan_z_s = z_s[nan_idx]
                print(f"nan z_s: {nan_z_s}")
                print(f"nan obs: {obs[nan_idx]}")
            except:
                pass
            raise ValueError("z is nan")
        z = z.cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def get_traj_meta(self, traj: np.ndarray) -> MetaDict:
        # traj: (traj_len, obs_dim)
        assert len(traj.shape) == 2
        obs = torch.as_tensor(traj).to(self.cfg.device)
        with torch.no_grad():
            obs = self.encoder(obs)
            z = self.feature_learner.feature_net(obs)
        # calcualte the z_diff by sliding window
        z_diff = torch.diff(z, dim=0)
        z_diff = math.sqrt(self.cfg.z_dim) * F.normalize(z_diff, dim=1)
        return z_diff.cpu().numpy(), z.cpu().numpy()

    def get_z_rewards(self, obs: np.ndarray, next_obs: np.ndarray, z_hilbert: np.ndarray, normalize: bool = True) -> np.ndarray:
        obs = torch.as_tensor(obs).to(self.cfg.device)
        next_obs = torch.as_tensor(next_obs).to(self.cfg.device)
        z_hilbert = torch.as_tensor(z_hilbert).to(self.cfg.device)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            z_next_obs = self.feature_learner.feature_net(next_obs)
            obs = self.encoder(obs)
            z_obs = self.feature_learner.feature_net(obs)
            z_diff = z_next_obs - z_obs
            diff_rewards = torch.einsum('sd, sd -> s', z_diff, z_hilbert)
            state_rewards = torch.einsum('sd, sd -> s', z_next_obs, z_hilbert)
        if normalize:
            diff_rewards = diff_rewards / math.sqrt(self.cfg.z_dim)
            state_rewards = state_rewards / math.sqrt(self.cfg.z_dim)
        return diff_rewards.squeeze(0).cpu().numpy(), state_rewards.squeeze(0).cpu().numpy()

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, feat_type: str = 'state'):
        with torch.no_grad():
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

        with torch.no_grad():
            if feat_type == 'state':
                phi = self.feature_learner.feature_net(obs)
            elif feat_type == 'diff':
                phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
        z = torch.linalg.lstsq(phi, reward).solution
        with torch.no_grad():
            r_vec = reward.view(-1)  # (N,)
            y_hat = phi @ z  # (N,)
            resid = r_vec - y_hat

            N, D = phi.shape
            sse = (resid ** 2).sum()
            rmse = torch.sqrt((resid ** 2).mean())
            # 避免除零
            denom = torch.clamp((r_vec - r_vec.mean()).pow(2).sum(), min=1e-12)
            r2 = 1.0 - sse / denom

            # 数值稳定性
            svals = torch.linalg.svdvals(phi)  # (min(N,D),)
            s_min = torch.clamp_min(svals.min(), 1e-12)
            cond = (svals.max() / s_min)
            # 有效秩（基于阈值）
            rank = int((svals > 1e-6).sum().item())

            # 置信区间（可选；N > D 时有意义）
            ci_low = None; ci_high = None
            if N > D:
                sigma2 = (sse / (N - D)).item()
                # (ΦᵀΦ)^{-1}
                xtx = phi.T @ phi
                cov = sigma2 * torch.linalg.pinv(xtx)
                se = torch.sqrt(torch.clamp(torch.diag(cov), min=0.0))
                ci_low = (z - 1.96 * se)
                ci_high = (z + 1.96 * se)

            diag = {
                "N": N, "D": D,
                "SSE": float(sse.item()),
                "RMSE": float(rmse.item()),
                "R2": float(r2.item()),
                "cond": float(cond.item()),
                "rank": rank,
            }
            # 如果你愿意，也可以把 ci 的范数或最大/最小值记录一下：
            if ci_low is not None:
                diag.update({
                    "z_se_max": float(se.max().item()),
                    "z_se_mean": float(se.mean().item()),
                })
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        return meta, diag

    def sample_z(self, size, env_command=None, obs=None, next_obs=None):
        if self.cfg.random_sample_z:
            z_hilbert = torch.randn((size, self.cfg.z_dim), dtype=torch.float32).to(self.cfg.device)
            z_hilbert = math.sqrt(self.cfg.z_dim) * F.normalize(z_hilbert, dim=1)
        else:
            z_hilbert = torch.as_tensor(self.get_goal_meta(next_obs, obs if self.cfg.feature_type == 'diff' else None)['z']).to(self.cfg.device)
        if self.command_injection:
            assert not self.use_raw_command
            z_actor = self.command_injection_net(env_command)
            z_actor = math.sqrt(self.cfg.z_dim) * F.normalize(z_actor, dim=1)
        elif self.use_raw_command:
            assert not self.command_injection
            z_actor = env_command
        else:
            z_actor = z_hilbert
        return z_hilbert, z_actor

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
            self,
            meta: MetaDict,
            global_step: int,
            time_step: TimeStep,
            finetune: bool = False,
            replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0:
            return self.init_meta()
        return meta

    @torch.no_grad()
    def act_inference(self, observations, z_vector) -> tp.Any:
        obs = torch.as_tensor(observations, device=self.cfg.device, dtype=torch.float32)
        h = self.encoder(obs)
        z = torch.as_tensor(z_vector, device=self.cfg.device)
        action = self.actor.act_inference(h, z)
        return action

    def update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int,
        is_train: bool = True
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        if not self.cfg.train_phi_only:
            with torch.no_grad():
                next_action, logp = self.actor.sample_and_logprob(next_obs, z)
                next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)  # batch x z_dim
                if self.cfg.feature_type == 'state':
                    target_phi = self.feature_learner.feature_net(next_obs).detach()  # batch x z_dim
                elif self.cfg.feature_type == 'diff':
                    target_phi = self.feature_learner.feature_net(next_obs).detach() - self.feature_learner.feature_net(obs).detach()
                else:
                    target_phi = torch.cat([self.feature_learner.feature_net(obs).detach(), self.feature_learner.feature_net(next_obs).detach()], dim=-1)
                next_Q1, next_Q2 = [torch.einsum('sd, sd -> s', next_Fi, z) for next_Fi in [next_F1, next_F2]]
                next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
                target_F = target_phi + discount * next_F

            F1, F2 = self.successor_net(obs, z, action)
            if self.cfg.q_loss:
                Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
                target_Q = torch.einsum('sd, sd -> s', target_F, z)
                sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            else:
                sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)

        # compute feature loss
        if self.cfg.feature_learner == 'hilp':
            phi_loss, info = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)
        else:
            phi_loss = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)
            info = None

        if self.cfg.use_tb or self.cfg.use_wandb:
            if not self.cfg.train_phi_only:
                metrics['target_F'] = target_F.mean().item()
                metrics['F1'] = F1.mean().item()
                metrics['phi'] = target_phi.mean().item()
                metrics['phi_norm'] = torch.norm(target_phi, dim=-1).mean().item()
                metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
                metrics['sf_loss'] = sf_loss.item()
                # 在update_sf方法中添加
                metrics['F1_norm'] = torch.norm(F1, dim=-1).mean().item()
                metrics['F2_norm'] = torch.norm(F2, dim=-1).mean().item()  
                metrics['target_F_norm'] = torch.norm(target_F, dim=-1).mean().item()
                metrics['next_F_norm'] = torch.norm(next_F, dim=-1).mean().item()
                metrics['target_F_std'] = target_F.std().item()  # 目标的方差
                metrics['Q1_Q2_diff'] = torch.abs(next_Q1 - next_Q2).mean().item()  # 双网络差异

            if isinstance(self.sf_opt, torch.optim.Adam):
                metrics["sf_opt_lr"] = self.sf_opt.param_groups[0]["lr"]

            if info is not None:
                for key, val in info.items():
                    metrics[key] = val.item()
        if is_train:
            if not self.cfg.train_phi_only:
                # optimize SF
                if self.encoder_opt is not None:
                    self.encoder_opt.zero_grad(set_to_none=True)
                self.sf_opt.zero_grad(set_to_none=True)
                sf_loss.backward()
                sf_grad = grad_norm_stats(self.successor_net, prefix='sf')
                metrics.update(sf_grad)
                self.sf_opt.step()
                if self.encoder_opt is not None:
                    self.encoder_opt.step()
                    encoder_grad = grad_norm_stats(self.encoder, prefix='encoder')
                    metrics.update(encoder_grad)
            if self.phi_opt is not None:
                self.phi_opt.zero_grad(set_to_none=True)
                phi_loss.backward(retain_graph=True)
                phi_grad = grad_norm_stats(self.feature_learner, prefix='phi')
                metrics.update(phi_grad)
                self.phi_opt.step()
        return metrics

    def update_actor(self, obs: torch.Tensor, z_hilbert: torch.Tensor, z_actor: torch.Tensor, step: int, privileged_obs: torch.Tensor, is_train: bool = True) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        action, log_prob = self.actor.sample_and_logprob(obs, z_actor)
        F1, F2 = self.successor_net(obs, z_hilbert, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z_hilbert)
        Q2 = torch.einsum('sd, sd -> s', F2, z_hilbert)
        Q = torch.min(Q1, Q2)
        # actor_loss = (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()
        actor_loss = (self.cfg.temp * log_prob - Q).mean()
        # total_loss = privileged_recon_loss + actor_loss
        if is_train:
            # optimize actor
            self.actor_opt.zero_grad(set_to_none=True)
            actor_grad = grad_norm_stats(self.actor, prefix='actor')
            metrics.update(actor_grad)
            actor_loss.backward()
            self.actor_opt.step()
            if self.command_injection:
                # 使用torch.norm计算梯度范数
                grad_norms = []
                for p in self.command_injection_net.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm(2).item())
                
                if grad_norms:
                    total_grad_norm = sum(grad_norms)
                    max_grad_norm = max(grad_norms)
                    avg_grad_norm = total_grad_norm / len(grad_norms)
                    
                    metrics['command_net_total_grad_norm'] = total_grad_norm
                    metrics['command_net_max_grad_norm'] = max_grad_norm
                    metrics['command_net_avg_grad_norm'] = avg_grad_norm
                    
                self.command_injection_opt.step()
                self.command_injection_opt.zero_grad(set_to_none=True)

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['Q1'] = Q1.mean().item()
            metrics['Q2'] = Q2.mean().item()
            metrics['Q'] = Q.mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int, is_val: bool = False) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        for _ in range(self.cfg.num_sf_updates):
            batch = replay_loader.sample(self.cfg.batch_size, is_val=is_val)
            # sample_end = time.time()
            # print(f"sample time: {sample_end - sample_start}")
            batch = batch.to(self.cfg.device)
            obs = batch.obs
            action = batch.action
            discount = batch.discount
            next_obs = batch.next_obs
            future_obs = batch.future_obs
            privileged_obs = batch.privileged_obs
            try:
                commands_obs = batch.commands
            except:
                commands_obs = None
            # print(obs.shape, action.shape, discount.shape, next_obs.shape, future_obs.shape, privileged_obs.shape)

            z_hilbert, z_actor = self.sample_z(self.cfg.batch_size, commands_obs)
            z_hilbert = z_hilbert.to(self.cfg.device)
            z_actor = z_actor.to(self.cfg.device)
            if not z_hilbert.shape[-1] == self.cfg.z_dim:
                raise RuntimeError("There's something wrong with the logic here")

            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
            future_obs = self.aug_and_encode(future_obs)
            next_obs = next_obs.detach()

            if self.cfg.mix_ratio > 0:
                perm = torch.randperm(self.cfg.batch_size)
                with torch.no_grad():
                    if self.cfg.feature_type == 'state':
                        desired_obs = next_obs[perm]
                        phi = self.feature_learner.feature_net(desired_obs)
                    elif self.cfg.feature_type == 'diff':
                        desired_obs = obs[perm]
                        desired_next_obs = next_obs[perm]
                        phi = self.feature_learner.feature_net(desired_next_obs) - self.feature_learner.feature_net(desired_obs)
                    else:
                        desired_obs = obs[perm]
                        desired_next_obs = next_obs[perm]
                        phi = torch.cat([self.feature_learner.feature_net(desired_obs), self.feature_learner.feature_net(desired_next_obs)], dim=-1)
                # compute inverse of cov of phi
                cov = torch.matmul(phi.T, phi) / phi.shape[0]
                inv_cov = torch.linalg.pinv(cov)

                mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
                with torch.no_grad():
                    new_z = phi[mix_idxs]

                new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
                new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
                z_hilbert[mix_idxs] = new_z

            metrics.update(self.update_sf(obs=obs, action=action, discount=discount, next_obs=next_obs, future_obs=future_obs, z=z_hilbert.detach(), step=step))
            if not self.cfg.train_phi_only:
                # update actor
                metrics.update(self.update_actor(obs.detach(), z_hilbert, z_actor, step, privileged_obs))

                # update critic target
                utils.soft_update_params(self.successor_net, self.successor_target_net, self.cfg.sf_target_tau)
            # sample_start = time.time()
            # print("Update SF time: ", sample_start - sample_end)

        return metrics


    def update_batch(self, batch, step: int, is_train: bool = True) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        
        obs = batch['obs'].to(self.cfg.device)
        action = batch['actions'].to(self.cfg.device)
        discount = batch['discount'].to(self.cfg.device)
        if len(discount.shape) == 1:
            discount = discount.unsqueeze(1)
        next_obs = batch['next_obs'].to(self.cfg.device)
        future_obs = batch['future_obs'].to(self.cfg.device)
        privileged_obs = batch['privileged_obs'].to(self.cfg.device)
        commands_obs = batch['commands'].to(self.cfg.device) if "commands" in batch else None
        # print(obs.shape, action.shape, discount.shape, next_obs.shape, future_obs.shape, privileged_obs.shape)
        BS = obs.shape[0]
        z_hilbert, z_actor = self.sample_z(BS, commands_obs, obs, next_obs)
        z_hilbert = z_hilbert.to(self.cfg.device)
        z_actor = z_actor.to(self.cfg.device)
        if not z_hilbert.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")

        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)
        future_obs = self.aug_and_encode(future_obs)
        next_obs = next_obs.detach()

        if self.cfg.mix_ratio > 0 and self.cfg.random_sample_z:
            perm = torch.randperm(BS)
            with torch.no_grad():
                if self.cfg.feature_type == 'state':
                    desired_obs = next_obs[perm]
                    phi = self.feature_learner.feature_net(desired_obs)
                elif self.cfg.feature_type == 'diff':
                    desired_obs = obs[perm]
                    desired_next_obs = next_obs[perm]
                    phi = self.feature_learner.feature_net(desired_next_obs) - self.feature_learner.feature_net(desired_obs)
                else:
                    desired_obs = obs[perm]
                    desired_next_obs = next_obs[perm]
                    phi = torch.cat([self.feature_learner.feature_net(desired_obs), self.feature_learner.feature_net(desired_next_obs)], dim=-1)
            # compute inverse of cov of phi
            cov = torch.matmul(phi.T, phi) / phi.shape[0]
            inv_cov = torch.linalg.pinv(cov)

            mix_idxs: tp.Any = np.where(np.random.uniform(size=BS) < self.cfg.mix_ratio)[0]
            with torch.no_grad():
                new_z = phi[mix_idxs]

            new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
            new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
            z_hilbert[mix_idxs] = new_z
            z_actor[mix_idxs] = new_z

        metrics.update(self.update_sf(obs=obs, action=action, discount=discount, next_obs=next_obs, future_obs=future_obs, z=z_hilbert.detach(), step=step, is_train=is_train))
        if not self.cfg.train_phi_only:
            # update actor
            metrics.update(self.update_actor(obs.detach(), z_hilbert, z_actor, step, privileged_obs, is_train=is_train))

            # update critic target
            utils.soft_update_params(self.successor_net, self.successor_target_net, self.cfg.sf_target_tau)
        # sample_start = time.time()
        # print("Update SF time: ", sample_start - sample_end)
        return metrics


@torch.no_grad()
def grad_norm_stats(
    model: torch.nn.Module,
    prefix: str,
    norm_type: float = 2.0,
    only_requires_grad: bool = True,
) -> Dict[str, Any]:
    """
    统计当前已计算的梯度的范数信息。
    - global_norm: 所有参数梯度拼接后的整体 p-范数（p=norm_type）
    - max_per_param_norm: 每个参数张量梯度的 p-范数中的最大值
    - max_abs_grad: 所有梯度条目中的最大绝对值（∞-范数层面）
    注意：若使用混合精度且用了 GradScaler，请在调用前先对优化器 unscale：
        scaler.unscale_(optimizer)
    Args:
        model: 含有 .named_parameters() 的模块
        norm_type: 范数类型，常用 2.0 或 np.inf
        include_per_param: 是否返回每个参数的范数字典（可能略慢）
        only_requires_grad: 只考虑 requires_grad=True 的参数

    Returns:
        dict，键包括：
            global_norm, max_per_param_norm, max_abs_grad
    """
    if isinstance(norm_type, float) and math.isinf(norm_type):
        norm_type = float('inf')

    max_per_param = 0.0
    max_abs_grad = 0.0

    # 按 p-范数聚合得到 global_norm
    # p == inf 时：global = max(abs(grad))
    # 否则：global = (sum(|g|^p))^(1/p)
    if norm_type == float('inf'):
        global_accum: Optional[torch.Tensor] = None  # 标量张量：当前最大值
    else:
        global_accum = torch.zeros((), device=next(model.parameters()).device)

    for name, p in model.named_parameters():
        if only_requires_grad and not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            continue

        # 每参数的 p-范数
        if norm_type == float('inf'):
            param_norm = g.detach().abs().max().item()
            # 更新 global_accum
            cur_max = g.detach().abs().max()
            global_accum = cur_max if global_accum is None else torch.maximum(global_accum, cur_max)
        else:
            # torch.linalg.vector_norm 在新版本更好；兼容性用 torch.norm
            param_norm = torch.norm(g.detach(), p=norm_type).item()
            # 累加 |g|^p
            global_accum = global_accum + torch.sum(g.detach().abs().pow(norm_type))

        # 追踪最大 per-param 范数
        if param_norm > max_per_param:
            max_per_param = float(param_norm)

        # 追踪最大绝对梯度条目（无关 p）
        max_abs_grad = max(max_abs_grad, g.detach().abs().max().item())

    # 汇总 global norm
    if norm_type == float('inf'):
        global_norm = float(global_accum.item() if global_accum is not None else 0.0)
    else:
        global_norm = float(global_accum.pow(1.0 / norm_type).item())

    out = {
        f"{prefix}_global_norm": global_norm,
        f"{prefix}_max_per_param_norm": max_per_param,
        f"{prefix}_max_abs_grad": max_abs_grad,
    }
    return out
