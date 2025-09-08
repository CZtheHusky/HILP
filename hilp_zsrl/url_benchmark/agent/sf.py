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

from url_benchmark import utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from .ddpg import MetaDict, make_aug_encoder
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, mlp, OnlineCov
from url_benchmark.dmc import TimeStep
from url_benchmark.hugwbc_policy_network import create_hugwbc_policy
import time

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SFAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.sf.SFAgent"
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
    use_large_phi_net: bool = False


cs = ConfigStore.instance()
cs.store(group="agent", name="sf", node=SFAgentConfig)


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
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, cfg, use_large_phi_net=False) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.z_dim = z_dim
        self.cfg = cfg

        if self.cfg.feature_type != 'concat':
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        if use_large_phi_net:
            layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "ntanh", hidden_dim, "relu", feature_dim]
        else:
            layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", feature_dim]

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
        if self.training:   # we shall not update parameters during evaluation
            utils.soft_update_params(self.phi1, self.target_phi1, 0.005)
            utils.soft_update_params(self.phi2, self.target_phi2, 0.005)
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


class SFAgent:

    def __init__(self, **kwargs: tp.Any):
        cfg = SFAgentConfig(**kwargs)
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
        self.actor = create_hugwbc_policy(z_dim=cfg.z_dim, proprio_dim=self.obs_dim // 5, clock_dim=0).to(cfg.device)
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
        self.feature_learner = learner(self.obs_dim, self.action_dim, cfg.z_dim, cfg.phi_hidden_dim, use_large_phi_net=cfg.use_large_phi_net, **extra_kwargs).to(cfg.device)

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

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_sf:
            names += ["successor_net", "feature_learner", "successor_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

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
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)
            z_obs = self.feature_learner.feature_net(obs)
            z_next_obs = self.feature_learner.feature_net(next_obs)
            z_diff = z_next_obs - z_obs
            rewards = torch.einsum('sd, sd -> s', z_diff, z_hilbert)
        if normalize:
            rewards = rewards / math.sqrt(self.cfg.z_dim)
        return rewards.squeeze(0).cpu().numpy()
        

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor):
        with torch.no_grad():
            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

        with torch.no_grad():
            if self.cfg.feature_type == 'state':
                phi = self.feature_learner.feature_net(obs)
            elif self.cfg.feature_type == 'diff':
                phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
            else:
                raise NotImplementedError("feature_type must be state or diff")
                phi = torch.cat([self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)], dim=-1)
        z = torch.linalg.lstsq(phi, reward).solution

        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        return meta

    def sample_z(self, size, env_command=None):
        z_hilbert = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z_hilbert = math.sqrt(self.cfg.z_dim) * F.normalize(z_hilbert, dim=1)
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

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        # if self.cfg.boltzmann:
        #     dist = self.actor(h, z)
        # else:
        #     stddev = utils.schedule(self.cfg.stddev_schedule, step)
        #     dist = self.actor(h, z, stddev)
        if eval_mode:
            action, mem = self.actor.act_inference(h, z)
        else:
            action = self.actor.act(h, z)
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

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
                # if self.cfg.boltzmann:
                #     dist = self.actor(next_obs, z)
                #     next_action = dist.sample()
                # else:
                #     stddev = utils.schedule(self.cfg.stddev_schedule, step)
                #     dist = self.actor(next_obs, z, stddev)
                #     next_action = dist.sample(clip=self.cfg.stddev_clip)
                self.actor.update_distribution(next_obs, z)
                dist = self.actor.distribution
                next_action = dist.sample()
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
            if phi_loss is not None:
                metrics['phi_loss'] = phi_loss.item()

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
                self.sf_opt.step()
                if self.encoder_opt is not None:
                    self.encoder_opt.step()
            if self.phi_opt is not None:
                self.phi_opt.zero_grad(set_to_none=True)
                phi_loss.backward(retain_graph=True)
            if self.phi_opt is not None:
                self.phi_opt.step()
        return metrics

    def update_actor(self, obs: torch.Tensor, z_hilbert: torch.Tensor, z_actor: torch.Tensor, step: int, privileged_obs: torch.Tensor, is_train: bool = True) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # if self.cfg.boltzmann:
        #     dist = self.actor(obs, z)
        #     action = dist.rsample()
        # else:
        #     stddev = utils.schedule(self.cfg.stddev_schedule, step)
        #     dist = self.actor(obs, z, stddev)
        #     action = dist.sample(clip=self.cfg.stddev_clip)
        self.actor.update_distribution(obs, z_actor, privileged_obs, sync_update=True)
        dist = self.actor.distribution
        action = dist.rsample()
        privileged_recon_loss = self.actor.actor.privileged_recon_loss
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.successor_net(obs, z_hilbert, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z_hilbert)
        Q2 = torch.einsum('sd, sd -> s', F2, z_hilbert)
        Q = torch.min(Q1, Q2)
        # actor_loss = (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()
        actor_loss = (self.cfg.temp * log_prob - Q).mean()
        total_loss = privileged_recon_loss + actor_loss
        if is_train:
            # optimize actor
            self.actor_opt.zero_grad(set_to_none=True)
            total_loss.backward()
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
            metrics['privileged_recon_loss'] = privileged_recon_loss.item()
            metrics['total_loss'] = total_loss.item()

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
        z_hilbert, z_actor = self.sample_z(BS, commands_obs)
        z_hilbert = z_hilbert.to(self.cfg.device)
        z_actor = z_actor.to(self.cfg.device)
        if not z_hilbert.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")

        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)
        future_obs = self.aug_and_encode(future_obs)
        next_obs = next_obs.detach()

        if self.cfg.mix_ratio > 0:
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

        metrics.update(self.update_sf(obs=obs, action=action, discount=discount, next_obs=next_obs, future_obs=future_obs, z=z_hilbert.detach(), step=step, is_train=is_train))
        if not self.cfg.train_phi_only:
            # update actor
            metrics.update(self.update_actor(obs.detach(), z_hilbert, z_actor, step, privileged_obs, is_train=is_train))

            # update critic target
            utils.soft_update_params(self.successor_net, self.successor_target_net, self.cfg.sf_target_tau)
        # sample_start = time.time()
        # print("Update SF time: ", sample_start - sample_end)

        return metrics
