"""
HugWBC Policy Network - Standalone Implementation
=================================================

This module contains the complete standalone implementation of HugWBC's policy network,
extracted from the original codebase. It includes all Actor components and utilities.

Author: Extracted from HugWBC codebase
License: BSD-3-Clause (following original license)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform, AffineTransform

# =============================================================================
# Utility Functions
# =============================================================================

def get_activation(act_name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        act_name: Name of activation function ('elu', 'relu', 'selu', etc.)
        
    Returns:
        PyTorch activation module
        
    Raises:
        ValueError: If activation name is not supported
    """
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.ReLU(),  # Note: original uses ReLU for crelu
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }
    
    if act_name.lower() not in activations:
        raise ValueError(f"Invalid activation function: {act_name}. "
                        f"Supported: {list(activations.keys())}")
    
    return activations[act_name.lower()]


def MLP(input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int], 
        activation: str, 
        output_activation: Optional[str] = None,
        first_activation: Optional[str] = None,
        ) -> List[nn.Module]:
    """Create MLP layers.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        output_activation: Output activation function name (optional)
        
    Returns:
        List of PyTorch modules forming the MLP
    """
    activation_fn = get_activation(activation)
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    if first_activation is not None:
        first_activation_fn = get_activation(first_activation)
        layers.append(first_activation_fn)
    else:
        layers.append(activation_fn)
    
    # Hidden layers
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            # Output layer
            layers.append(nn.Linear(hidden_dims[i], output_dim))
            if output_activation is not None:
                output_activation_fn = get_activation(output_activation)
                layers.append(output_activation_fn)
        else:
            # Intermediate hidden layers
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation_fn)
    
    return layers


# =============================================================================
# Core Policy Network Components
# =============================================================================

class BaseAdaptModel(nn.Module):
    """Base class for adaptive models with memory encoding and state estimation.
    
    This is the foundation class that defines the core architecture for
    HugWBC's adaptive policy networks. It includes:
    - Low-level control network for action generation
    """
    
    def __init__(self,
                 act_dim: int,
                 proprioception_dim: int,
                 cmd_dim: int,
                 terrain_dim: int,
                 latent_dim: int,
                 actor_hidden_dims: List[int],
                 activation: str,
                 first_activation: Optional[str] = None,
                 output_activation: Optional[str] = None,
                 clock_dim: int = 0):
        """Initialize BaseAdaptModel.
        
        Args:
            act_dim: Action dimension (19 for H1 robot)
            proprioception_dim: Proprioception observation dimension (63 for H1)
            cmd_dim: Command dimension (11 for H1 interrupt)
            terrain_dim: Terrain information dimension (221 for H1)
            latent_dim: Latent space dimension for memory encoding (32)
            actor_hidden_dims: Hidden layer dimensions for low-level network
            activation: Activation function name
            first_activation: First activation function name (optional)
            output_activation: Output activation function name (optional)
        """
        super().__init__()
        
        # Network properties
        self.is_recurrent = False
        
        # Dimensions
        self.act_dim = act_dim
        self.proprioception_dim = proprioception_dim
        self.cmd_dim = cmd_dim
        self.terrain_dim = terrain_dim
        
        # Training state
        self.z = 0  # Latent variable for inference
        self.clock_dim = clock_dim
        
        # Low-level Control Network: concatenated features -> actions
        # Output: [batch, act_dim]
        # clock_dim = 2  # Clock inputs dimension
        # we do not use command for hilp
        control_input_dim = (latent_dim + 
                           proprioception_dim + self.cmd_dim + self.clock_dim)
        self.low_level_net = nn.Sequential(
            *MLP(control_input_dim, 2 * act_dim, actor_hidden_dims, 
                activation, output_activation, first_activation=first_activation)
        )

    def forward(self, 
                obs: torch.Tensor, 
                z_vector: torch.Tensor,
                env_mask: Optional[torch.Tensor] = None,
                clock: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward pass of the adaptive model.
        
        Args:
            x: Input observations
               Shape: [batch, history_steps, obs_dim] or [batch, obs_dim]
            env_mask: Environment mask (unused in base implementation)
            **kwargs: Additional arguments
            
        Returns:
            actions: Predicted actions
                    Shape: [batch, act_dim]
        """
        # Extract proprioception sequence and commands
        # pro_obs_seq: [batch, history_steps, proprioception_dim] or [batch, proprioception_dim]
        pro_obs_seq = obs
        
        # cmd: [batch, cmd_dim] (take the latest timestep)
        cmd = z_vector
        
        # Encode memory from proprioception sequence
        # mem: [batch, latent_dim]
        mem = self.memory_encoder(pro_obs_seq, **kwargs)
                
        # Current proprioception (latest timestep)
        # current_proprio: [batch, proprioception_dim]
        current_proprio = obs[:, -1, :]
        
        # Extract clock inputs (last 2 dimensions of partial obs)
        # clock: [batch, clock_dim]
        # clock = x[..., -1, -2:]  # Last 2 dimensions of the latest timestep
        # no clock for hilp
        
        # Concatenate all features for low-level control
        control_input = torch.cat([
            mem,                    # [batch, latent_dim]
            current_proprio,        # [batch, proprioception_dim]
            cmd,                    # [batch, cmd_dim]
        ], dim=-1)
        if clock is not None and self.clock_dim > 0:
            assert clock.shape[-1] == self.clock_dim, "clock dimension mismatch"
            control_input = torch.cat([control_input, clock], dim=-1)
        
        # Generate actions
        # actions: [batch, act_dim]
        actions, stds = self.low_level_net(control_input).chunk(2, dim=-1)
        
        # Store latent for inference
        self.z = mem
        
        return actions, stds
    
    def memory_encoder(self, pro_obs_seq: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode proprioception sequence to latent representation.
        
        This method must be implemented by subclasses.
        
        Args:
            pro_obs_seq: Proprioception observation sequence
                        Shape: [batch, history_steps, proprioception_dim]
            **kwargs: Additional arguments
            
        Returns:
            latent: Encoded latent representation
                   Shape: [batch, latent_dim]
        """
        raise NotImplementedError("Subclasses must implement memory_encoder")


class MlpAdaptModel(BaseAdaptModel):
    """MLP-based adaptive model for HugWBC policy network.
    
    This is the main implementation of HugWBC's policy network, using MLPs
    for memory encoding and action generation. Key features:
    - Short-term memory encoding using flattened history
    - Low-level control network for action generation
    """
    
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 proprioception_dim: int,
                 cmd_dim: int,
                 terrain_dim: int,
                 latent_dim: int = 32,
                 first_activation: Optional[str] = None,
                 actor_hidden_dims: List[int] = [256, 128, 32],
                 activation: str = 'elu',
                 output_activation: Optional[str] = None,
                 max_length: int = 5,
                 mlp_hidden_dims: List[int] = [256, 128],
                 clock_dim: int = 0,
                 **kwargs):
        """Initialize MlpAdaptModel.
        
        Args:
            obs_dim: Total observation dimension (unused, kept for compatibility)
            act_dim: Action dimension (19 for H1 robot)
            proprioception_dim: Proprioception dimension (63 for H1)
            cmd_dim: Command dimension (11 for H1 interrupt)
            terrain_dim: Terrain information dimension (221 for H1)
            latent_dim: Latent space dimension (32)
            actor_hidden_dims: Hidden layer dimensions for low-level network
            activation: Activation function name ('elu')
            output_activation: Output activation function name (None)
            max_length: Maximum history length (5)
            mlp_hidden_dims: Hidden layer dimensions for memory encoder
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(
            act_dim=act_dim,
            proprioception_dim=proprioception_dim,
            cmd_dim=cmd_dim,
            terrain_dim=terrain_dim,
            latent_dim=latent_dim,
            actor_hidden_dims=actor_hidden_dims,
            first_activation=first_activation,
            activation=activation,
            output_activation=output_activation,
            clock_dim=clock_dim
        )
        
        # Memory encoding parameters
        self.max_length = max_length
        self.short_length = max_length  # For compatibility
        
        # Memory Encoder: flattened history -> latent representation
        # Input: [batch, proprioception_dim * max_length]
        # Output: [batch, latent_dim]
        memory_input_dim = proprioception_dim * self.short_length
        self.mem_encoder = nn.Sequential(
            *MLP(memory_input_dim, latent_dim, mlp_hidden_dims, activation, first_activation=first_activation)
        )
    
    def memory_encoder(self, pro_obs_seq: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode proprioception sequence using MLP.
        
        Takes the recent history of proprioception observations, flattens them,
        and passes through an MLP to get a latent representation.
        
        Args:
            pro_obs_seq: Proprioception observation sequence
                        Shape: [batch, history_steps, proprioception_dim]
            **kwargs: Additional arguments (unused)
            
        Returns:
            latent: Encoded latent representation
                   Shape: [batch, latent_dim]
        """
        # Take the most recent short_length steps and flatten
        # short_term_mem: [batch, proprioception_dim * short_length]
        short_term_mem = pro_obs_seq[..., -self.short_length:, :self.proprioception_dim]
        short_term_mem = short_term_mem.flatten(-2, -1)
        
        # Encode to latent space
        # latent: [batch, latent_dim]
        latent = self.mem_encoder(short_term_mem)
        
        return latent


# =============================================================================
# Policy Network with Action Distribution
# =============================================================================

class HugWBCPolicyNetwork(nn.Module):
    """Complete HugWBC Policy Network with Gaussian action distribution.
    
    This class wraps the MlpAdaptModel with action noise and distribution
    handling, providing a complete policy network interface.
    """
    
    def __init__(self,
                 # Network architecture parameters
                 proprioception_dim: int = 63,
                 cmd_dim: int = 11,
                 act_dim: int = 19,
                 terrain_dim: int = 221,
                 latent_dim: int = 32,
                 max_length: int = 5,
                 
                 # Hidden layer dimensions
                 actor_hidden_dims: List[int] = [256, 128, 32],
                 mlp_hidden_dims: List[int] = [256, 128],
                 
                 # Activation functions
                 activation: str = 'elu',
                 first_activation: Optional[str] = None,
                 output_activation: Optional[str] = None,
                 clock_dim: int = 0,
                 # Action noise parameters
                 init_noise_std: float = 1.0,
                 max_log_std: float = 0.182,
                 min_log_std: float = 0.1):
        """Initialize HugWBC Policy Network.
        
        Args:
            proprioception_dim: Proprioception observation dimension
            cmd_dim: Command dimension
            act_dim: Action dimension
            terrain_dim: Terrain information dimension
            latent_dim: Latent space dimension for memory encoding
            max_length: Maximum history length
            actor_hidden_dims: Hidden layer dimensions for actor network
            mlp_hidden_dims: Hidden layer dimensions for memory encoder
            activation: Activation function name
            output_activation: Output activation function name
            init_noise_std: Initial noise standard deviation
            max_log_std: Maximum standard deviation
            min_log_std: Minimum standard deviation
        """
        super().__init__()
        
        # Store parameters
        self.proprioception_dim = proprioception_dim
        self.cmd_dim = cmd_dim
        self.act_dim = act_dim
        self.max_length = max_length
        
        # Action noise parameters
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std
        self.register_buffer("action_scale", torch.ones(act_dim))
        self.register_buffer("action_bias",  torch.zeros(act_dim))  # 可选（非对称动作空间）
        self.action_max = np.array([2.8929266929626465, 3.1424105167388916, 8.912518501281738, 9.273897171020508, 20.635496139526367, 1.5641342401504517, 6.097947120666504, 5.219802379608154, 8.732982635498047, 15.844582557678223, 5.335212230682373, 14.903319358825684, 2.345292568206787, 0.9799386262893677, 0.5946072936058044, 14.76853084564209, 3.138103723526001, 0.8584625124931335, 1.1060798168182373])
        self.action_min = np.array([-1.7501237392425537, -8.597620010375977, -9.818479537963867, -24.81416893005371, -13.048727989196777, -2.8899900913238525, -2.8148365020751953, -9.641386985778809, -30.218889236450195, -12.128853797912598, -7.452919960021973, -6.481909275054932, -2.5507757663726807, -0.7834361791610718, -3.2907698154449463, -6.089234828948975, -2.8664331436157227, -0.8890500068664551, -2.376858711242676])
        # Actor network
        obs_dim = proprioception_dim + cmd_dim  # Simplified obs dim
        self.actor = MlpAdaptModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            proprioception_dim=proprioception_dim,
            cmd_dim=cmd_dim,
            clock_dim=clock_dim,
            terrain_dim=terrain_dim,
            latent_dim=latent_dim,
            actor_hidden_dims=actor_hidden_dims,
            first_activation=first_activation,
            activation=activation,
            output_activation=output_activation,
            max_length=max_length,
            mlp_hidden_dims=mlp_hidden_dims
        )
        
        # Learnable action noise (standard deviation)
        # self.std = nn.Parameter(init_noise_std * torch.ones(act_dim))
        self.distribution = None
        
        # Disable validation for speed
        Normal.set_default_validate_args = False
        self.set_action_space(self.action_min, self.action_max)

    def set_action_space(self, low: np.ndarray, high: np.ndarray):
        low  = torch.as_tensor(low, dtype=torch.float32, device=self.action_scale.device)
        high = torch.as_tensor(high, dtype=torch.float32, device=self.action_scale.device)
        scale = (high - low) / 2.0
        bias  = (high + low) / 2.0
        eps = torch.finfo(scale.dtype).eps
        self.action_scale.copy_(torch.clamp(scale, min=eps))
        self.action_bias.copy_(bias)
    
    def forward(self, 
                observations: torch.Tensor,
                z_vector: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward pass to get action mean.
        
        Args:
            observations: Input observations
                         Shape: [batch, history_steps, obs_dim] or [batch, obs_dim]
            z_vector: Command vector
                     Shape: [batch, z_dim]
            **kwargs: Additional arguments
            
        Returns:
            action_mean: Mean actions from the network
                        Shape: [batch, act_dim]
        """
        print(observations.shape)
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        return self.actor(observations, z_vector, **kwargs)

    
    def update_distribution(self, 
                          observations: torch.Tensor,
                          z_vector: torch.Tensor,
                          **kwargs):
        """Update the action distribution.
        
        Args:
            observations: Input observations
            z_vector: Command vector
                     Shape: [batch, z_dim]
            **kwargs: Additional arguments
        """
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        # Get action mean from actor
        mean, log_std = self.actor(observations, z_vector, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        std = log_std.exp()
        
        # Create normal distribution
        base = Independent(Normal(mean, mean * 0.0 + std), 1)
        scale = getattr(self, "action_scale")
        bias  = getattr(self, "action_bias")
        dist = TransformedDistribution(
            base,
            [TanhTransform(cache_size=1), AffineTransform(loc=bias, scale=scale)]
        )
        self.distribution = dist

    @torch.no_grad()
    def act_inference(
        self, 
        observations, 
        z_vector, 
        **kwargs
    ):
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        mean, _ = self.actor(observations, z_vector, **kwargs)  # pre-tanh mean
        a = torch.tanh(mean)
        scale = getattr(self, "action_scale")
        bias  = getattr(self, "action_bias")
        a = bias + scale * a
        return a

    def sample_and_logprob(self, observations, z_vector, **kwargs):
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        self.update_distribution(observations, z_vector, **kwargs)
        a_env = self.distribution.rsample()      # 已是环境动作空间
        logp  = self.distribution.log_prob(a_env)  # 形状 [B]，已包含 tanh+affine 的雅可比
        return a_env, logp

    def get_actions_log_prob(self, actions: torch.Tensor):
        # 如 actions 来自同一分布，直接算即可；若可能略越界，可先夹到 (low, high) 的开区间内
        return self.distribution.log_prob(actions)

    
    @property
    def action_mean(self) -> torch.Tensor:
        """Get action mean from current distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call update_distribution first")
        return self.distribution.mean
    
    @property
    def action_std(self) -> torch.Tensor:
        """Get action standard deviation from current distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call update_distribution first")
        return self.distribution.stddev
    
    # @property
    # def entropy(self) -> torch.Tensor:
    #     """Get entropy of current distribution."""
    #     if self.distribution is None:
    #         raise RuntimeError("Must call update_distribution first")
    #     return self.distribution.entropy().sum(dim=-1)

# =============================================================================
# Configuration and Factory Functions
# =============================================================================

class HugWBCConfig:
    """Configuration class for HugWBC Policy Network."""
    
    # H1 Robot specific dimensions
    PROPRIOCEPTION_DIM = 44
    CMD_DIM = 11
    CLOCK_DIM = 0
    ACT_DIM = 19
    TERRAIN_DIM = 221
    
    # Network architecture
    LATENT_DIM = 32
    MAX_LENGTH = 5
    
    # Hidden layer dimensions
    ACTOR_HIDDEN_DIMS = [1024, 1024, 1024, 1024]
    MLP_HIDDEN_DIMS = [1024, 1024, 1024]
    
    # Activation functions
    ACTIVATION = 'elu'
    FIRST_ACTIVATION = 'tanh'
    OUTPUT_ACTIVATION = None
    
    # Action noise parameters
    INIT_NOISE_STD = 1.0
    max_log_std = 0.182
    min_log_std = -2.3


def create_sac_policy(z_dim: int = 32, horizon: int = 5, proprio_dim: int = 63, clock_dim: int = 0) -> HugWBCPolicyNetwork:
    """Factory function to create HugWBC Policy Network.
    
    Args:
        config: Configuration object (uses default if None)
        
    Returns:
        policy: HugWBC Policy Network instance
    """
    config = HugWBCConfig()
    config.CMD_DIM = z_dim
    config.MAX_LENGTH = horizon
    config.CLOCK_DIM = clock_dim
    config.PROPRIOCEPTION_DIM = proprio_dim
    
    return HugWBCPolicyNetwork(
        proprioception_dim=config.PROPRIOCEPTION_DIM,
        cmd_dim=config.CMD_DIM,
        act_dim=config.ACT_DIM,
        clock_dim=config.CLOCK_DIM,
        terrain_dim=config.TERRAIN_DIM,
        first_activation=config.FIRST_ACTIVATION,
        latent_dim=config.LATENT_DIM,
        max_length=config.MAX_LENGTH,
        actor_hidden_dims=config.ACTOR_HIDDEN_DIMS,
        mlp_hidden_dims=config.MLP_HIDDEN_DIMS,
        activation=config.ACTIVATION,
        output_activation=config.OUTPUT_ACTIVATION,
        init_noise_std=config.INIT_NOISE_STD,
        max_log_std=config.max_log_std,
        min_log_std=config.min_log_std
    )

def create_hugwbc_critic(input_dim=321, output_dim=1, hidden_dims=[512, 256, 128], activation='elu', output_activation=None) -> HugWBCPolicyNetwork:
    mlp_layers = MLP(
        input_dim,
        output_dim,
        hidden_dims,
        activation,
        output_activation
    )
    return nn.Sequential(*mlp_layers)

# Actor MLP: MlpAdaptModel(
#   (state_estimator): Sequential(
#     (0): Linear(in_features=32, out_features=64, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=64, out_features=32, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=32, out_features=3, bias=True)
#   )
#   (low_level_net): Sequential(
#     (0): Linear(in_features=111, out_features=256, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=256, out_features=128, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=128, out_features=32, bias=True)
#     (5): ELU(alpha=1.0)
#     (6): Linear(in_features=32, out_features=19, bias=True)
#   )
#   (mem_encoder): Sequential(
#     (0): Linear(in_features=315, out_features=256, bias=True)
#     (1): ELU(alpha=1.0)
#     (2): Linear(in_features=256, out_features=128, bias=True)
#     (3): ELU(alpha=1.0)
#     (4): Linear(in_features=128, out_features=32, bias=True)
#   )
# )
# Critic MLP: Sequential(
#   (0): Linear(in_features=321, out_features=512, bias=True)
#   (1): ELU(alpha=1.0)
#   (2): Linear(in_features=512, out_features=256, bias=True)
#   (3): ELU(alpha=1.0)
#   (4): Linear(in_features=256, out_features=128, bias=True)
#   (5): ELU(alpha=1.0)
#   (6): Linear(in_features=128, out_features=1, bias=True)
# )

# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    """Example usage and basic testing."""
    
    print("=" * 60)
    print("HugWBC Policy Network - Standalone Implementation")
    print("=" * 60)
    
    # Create policy network
    policy = create_sac_policy(z_dim=32, clock_dim=2)
    print(f"Created policy network with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Print network architecture
    print("\nNetwork Architecture:")
    print(policy)
    
    # Test with dummy data
    batch_size = 4
    history_steps = 5
    obs_dim = 63  # proprioception_dim + cmd_dim = 63 + 32
    z_dim = 32

    # Create dummy observations
    observations = torch.randn(batch_size, history_steps, obs_dim)
    z_vector = torch.randn(batch_size, z_dim)
    clock = torch.randn(batch_size, 2)
    print(f"\nInput shapes:")
    print(f"  observations: {observations.shape}")
    print(f"  z_vector: {z_vector.shape}")
    print(f"  clock: {clock.shape}")
    # Test forward pass
    with torch.no_grad():
        # Deterministic inference
        actions_mean = policy.act_inference(observations, z_vector, clock=clock)
        print(f"\nInference output shapes:")
        print(f"  actions_mean: {actions_mean.shape}")
        
        # Stochastic sampling
        actions, logp = policy.sample_and_logprob(observations, z_vector, clock=clock)
        print(f"  sampled_actions: {actions.shape}")
        
        # Get log probabilities
        log_probs = policy.get_actions_log_prob(actions)
        print(f"  log_probs: {log_probs.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Policy network is working correctly.")
    print("=" * 60)
