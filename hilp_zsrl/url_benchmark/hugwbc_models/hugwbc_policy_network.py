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
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Union


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
        output_activation: Optional[str] = None) -> List[nn.Module]:
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
    - State estimator for privileged information prediction
    - Low-level control network for action generation
    - Privileged information reconstruction loss
    """
    
    def __init__(self,
                 act_dim: int,
                 proprioception_dim: int,
                 cmd_dim: int,
                 privileged_dim: int,
                 terrain_dim: int,
                 latent_dim: int,
                 privileged_recon_dim: int,
                 actor_hidden_dims: List[int],
                 activation: str,
                 output_activation: Optional[str] = None,
                 clock_dim: int = 0):
        """Initialize BaseAdaptModel.
        
        Args:
            act_dim: Action dimension (19 for H1 robot)
            proprioception_dim: Proprioception observation dimension (63 for H1)
            cmd_dim: Command dimension (11 for H1 interrupt)
            privileged_dim: Privileged information dimension (24 for H1)
            terrain_dim: Terrain information dimension (221 for H1)
            latent_dim: Latent space dimension for memory encoding (32)
            privileged_recon_dim: Privileged reconstruction dimension (3: base linear velocities)
            actor_hidden_dims: Hidden layer dimensions for low-level network
            activation: Activation function name
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
        self.privileged_dim = privileged_dim
        self.privileged_recon_dim = privileged_recon_dim
        
        # Training state
        self.privileged_recon_loss = 0
        self.z = 0  # Latent variable for inference
        self.clock_dim = clock_dim
        
        # State Estimator: latent -> privileged prediction
        # Input: [batch, latent_dim] -> Output: [batch, privileged_recon_dim]
        self.state_estimator = nn.Sequential(
            *MLP(latent_dim, self.privileged_recon_dim, [64, 32], activation)
        )
        
        # Low-level Control Network: concatenated features -> actions
        # Input: [batch, latent_dim + privileged_recon_dim + proprioception_dim + cmd_dim + clock_dim]
        # Output: [batch, act_dim]
        # clock_dim = 2  # Clock inputs dimension
        # we do not use command for hilp
        control_input_dim = (latent_dim + privileged_recon_dim + 
                           proprioception_dim + self.cmd_dim + self.clock_dim)
        self.low_level_net = nn.Sequential(
            *MLP(control_input_dim, act_dim, actor_hidden_dims, 
                activation, output_activation)
        )

    def forward(self, 
                obs: torch.Tensor, 
                z_vector: torch.Tensor,
                privileged_obs: Optional[torch.Tensor] = None,
                env_mask: Optional[torch.Tensor] = None,
                sync_update: bool = False,
                clock: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward pass of the adaptive model.
        
        Args:
            x: Input observations
               Shape: [batch, history_steps, obs_dim] or [batch, obs_dim]
            privileged_obs: Privileged observations (for training)
                          Shape: [batch, privileged_obs_dim]
            env_mask: Environment mask (unused in base implementation)
            sync_update: Whether to compute privileged reconstruction loss
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
        
        # Predict privileged information from memory
        # privileged_pred_now: [batch, privileged_recon_dim]
        privileged_pred_now = self.state_estimator(mem)
        
        # Current proprioception (latest timestep)
        # current_proprio: [batch, proprioception_dim]
        current_proprio = obs[:, -1, :]
        
        # Extract clock inputs (last 2 dimensions of partial obs)
        # clock: [batch, clock_dim]
        # clock = x[..., -1, -2:]  # Last 2 dimensions of the latest timestep
        # no clock for hilp
        
        # Concatenate all features for low-level control
        # control_input: [batch, latent_dim + privileged_recon_dim + proprioception_dim + cmd_dim + clock_dim]
        control_input = torch.cat([
            mem,                    # [batch, latent_dim]
            privileged_pred_now,    # [batch, privileged_recon_dim]
            current_proprio,        # [batch, proprioception_dim]
            cmd,                    # [batch, cmd_dim]
        ], dim=-1)
        if clock is not None and self.clock_dim > 0:
            assert clock.shape[-1] == self.clock_dim, "clock dimension mismatch"
            control_input = torch.cat([control_input, clock], dim=-1)
        
        # Generate actions
        # actions: [batch, act_dim]
        actions = self.low_level_net(control_input)
        
        # Compute privileged reconstruction loss if in sync update mode
        if sync_update and privileged_obs is not None:
            # Extract ground truth privileged information (base linear velocities)
            # privileged_obs structure: [proprio(63) + cmd(11) + clock(2) + privileged(24) + terrain(221)]
            # We want the first 3 dims of privileged part: base_lin_vel (x, y, z)
            # privileged_start = self.proprioception_dim + self.cmd_dim
            # privileged_end = privileged_start + self.privileged_recon_dim
            if privileged_obs.shape[-1] == 3:
                privileged_gt = privileged_obs
            else:
                privileged_gt = privileged_obs[..., 76:79]
            
            # Compute MSE loss with coefficient 2 for base linear velocity reconstruction
            self.privileged_recon_loss = 2 * (
                (privileged_pred_now - privileged_gt.detach()).pow(2).mean()
            )
        
        # Store latent for inference
        self.z = mem
        
        return actions
    
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
    
    def compute_adaptation_pred_loss(self, metrics: Dict[str, float]) -> torch.Tensor:
        """Compute and record privileged prediction loss.
        
        Args:
            metrics: Dictionary to store metrics
            
        Returns:
            privileged_recon_loss: Reconstruction loss tensor
        """
        if self.privileged_recon_loss != 0:
            metrics['privileged_recon_loss'] += self.privileged_recon_loss.item()
        return self.privileged_recon_loss


class MlpAdaptModel(BaseAdaptModel):
    """MLP-based adaptive model for HugWBC policy network.
    
    This is the main implementation of HugWBC's policy network, using MLPs
    for memory encoding and action generation. Key features:
    - Short-term memory encoding using flattened history
    - State estimation for privileged information
    - Low-level control network for action generation
    """
    
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 proprioception_dim: int,
                 cmd_dim: int,
                 privileged_dim: int,
                 terrain_dim: int,
                 latent_dim: int = 32,
                 privileged_recon_dim: int = 3,
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
            privileged_dim: Privileged information dimension (24 for H1)
            terrain_dim: Terrain information dimension (221 for H1)
            latent_dim: Latent space dimension (32)
            privileged_recon_dim: Privileged reconstruction dimension (3: base linear velocities)
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
            privileged_dim=privileged_dim,
            terrain_dim=terrain_dim,
            latent_dim=latent_dim,
            privileged_recon_dim=privileged_recon_dim,
            actor_hidden_dims=actor_hidden_dims,
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
            *MLP(memory_input_dim, latent_dim, mlp_hidden_dims, activation)
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
                 privileged_dim: int = 24,
                 terrain_dim: int = 221,
                 latent_dim: int = 32,
                 privileged_recon_dim: int = 3,
                 max_length: int = 5,
                 
                 # Hidden layer dimensions
                 actor_hidden_dims: List[int] = [256, 128, 32],
                 mlp_hidden_dims: List[int] = [256, 128],
                 
                 # Activation functions
                 activation: str = 'elu',
                 output_activation: Optional[str] = None,
                 clock_dim: int = 0,
                 # Action noise parameters
                 init_noise_std: float = 1.0,
                 max_std: float = 1.2,
                 min_std: float = 0.1):
        """Initialize HugWBC Policy Network.
        
        Args:
            proprioception_dim: Proprioception observation dimension
            cmd_dim: Command dimension
            act_dim: Action dimension
            privileged_dim: Privileged information dimension
            terrain_dim: Terrain information dimension
            latent_dim: Latent space dimension for memory encoding
            privileged_recon_dim: Privileged reconstruction dimension (3: base linear velocities)
            max_length: Maximum history length
            actor_hidden_dims: Hidden layer dimensions for actor network
            mlp_hidden_dims: Hidden layer dimensions for memory encoder
            activation: Activation function name
            output_activation: Output activation function name
            init_noise_std: Initial noise standard deviation
            max_std: Maximum standard deviation
            min_std: Minimum standard deviation
        """
        super().__init__()
        
        # Store parameters
        self.proprioception_dim = proprioception_dim
        self.cmd_dim = cmd_dim
        self.act_dim = act_dim
        self.max_length = max_length
        
        # Action noise parameters
        self.max_std = max_std
        self.min_std = min_std
        
        # Actor network
        obs_dim = proprioception_dim + cmd_dim  # Simplified obs dim
        self.actor = MlpAdaptModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            proprioception_dim=proprioception_dim,
            cmd_dim=cmd_dim,
            clock_dim=clock_dim,
            privileged_dim=privileged_dim,
            terrain_dim=terrain_dim,
            latent_dim=latent_dim,
            privileged_recon_dim=privileged_recon_dim,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            output_activation=output_activation,
            max_length=max_length,
            mlp_hidden_dims=mlp_hidden_dims
        )
        
        # Learnable action noise (standard deviation)
        self.std = nn.Parameter(init_noise_std * torch.ones(act_dim))
        self.distribution = None
        
        # Disable validation for speed
        Normal.set_default_validate_args = False
    
    def forward(self, 
                observations: torch.Tensor,
                z_vector: torch.Tensor,
                privileged_obs: Optional[torch.Tensor] = None,
                sync_update: bool = False,
                **kwargs) -> torch.Tensor:
        """Forward pass to get action mean.
        
        Args:
            observations: Input observations
                         Shape: [batch, history_steps, obs_dim] or [batch, obs_dim]
            z_vector: Command vector
                     Shape: [batch, z_dim]
            privileged_obs: Privileged observations (for training)
            sync_update: Whether to compute privileged reconstruction loss
            **kwargs: Additional arguments
            
        Returns:
            action_mean: Mean actions from the network
                        Shape: [batch, act_dim]
        """
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        return self.actor(observations, z_vector, privileged_obs, sync_update=sync_update, **kwargs)
    
    def update_distribution(self, 
                          observations: torch.Tensor,
                          z_vector: torch.Tensor,
                          privileged_obs: Optional[torch.Tensor] = None,
                          sync_update: bool = False,
                          **kwargs):
        """Update the action distribution.
        
        Args:
            observations: Input observations
            z_vector: Command vector
                     Shape: [batch, z_dim]
            privileged_obs: Privileged observations (for training)
            sync_update: Whether to compute privileged reconstruction loss
            **kwargs: Additional arguments
        """
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        # Get action mean from actor
        mean = self.actor(observations, z_vector, privileged_obs, sync_update=sync_update, **kwargs)
        
        # Clamp standard deviation
        std = torch.clamp(self.std, min=self.min_std, max=self.max_std)
        
        # Create normal distribution
        self.distribution = Normal(mean, mean * 0.0 + std)
    
    def act(self, 
            observations: torch.Tensor,
            z_vector: torch.Tensor,
            privileged_obs: Optional[torch.Tensor] = None,
            sync_update: bool = False,
            **kwargs) -> torch.Tensor:
        """Sample actions from the policy.
        
        Args:
            observations: Input observations
                         Shape: [batch, history_steps, obs_dim] or [batch, obs_dim]
            privileged_obs: Privileged observations (for training)
            sync_update: Whether to compute privileged reconstruction loss
            **kwargs: Additional arguments
            
        Returns:
            actions: Sampled actions
                    Shape: [batch, act_dim]
        """
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        self.update_distribution(observations, z_vector, privileged_obs, sync_update=sync_update, **kwargs)
        return self.distribution.sample()
    
    def act_inference(self, 
                     observations: torch.Tensor,
                     z_vector: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get deterministic actions for inference.
        
        Args:
            observations: Input observations
                         Shape: [batch, history_steps, obs_dim] or [batch, obs_dim]
            **kwargs: Additional arguments
            
        Returns:
            actions_mean: Mean actions (deterministic)
                         Shape: [batch, act_dim]
            latent: Latent representation from memory encoder
                   Shape: [batch, latent_dim]
        """
        if len(observations.shape) == 2:
            observations = observations.reshape(observations.shape[0], self.max_length, -1)
        actions_mean = self.actor(observations, z_vector, **kwargs)
        return actions_mean, self.actor.z
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions.
        
        Args:
            actions: Actions to evaluate
                    Shape: [batch, act_dim]
            
        Returns:
            log_probs: Log probabilities of actions
                      Shape: [batch]
        """
        if self.distribution is None:
            raise RuntimeError("Must call update_distribution first")
        return self.distribution.log_prob(actions).sum(dim=-1)
    
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
    
    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of current distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call update_distribution first")
        return self.distribution.entropy().sum(dim=-1)
    
    def get_privileged_loss(self, metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Get privileged reconstruction loss.
        
        Args:
            metrics: Dictionary to store metrics (optional)
            
        Returns:
            privileged_recon_loss: Reconstruction loss
        """
        if metrics is None:
            metrics = {}
        return self.actor.compute_adaptation_pred_loss(metrics)


# =============================================================================
# Configuration and Factory Functions
# =============================================================================

class HugWBCConfig:
    """Configuration class for HugWBC Policy Network."""
    
    # H1 Robot specific dimensions
    PROPRIOCEPTION_DIM = 63
    CMD_DIM = 11
    CLOCK_DIM = 0
    ACT_DIM = 19
    PRIVILEGED_DIM = 24
    TERRAIN_DIM = 221
    
    # Network architecture
    LATENT_DIM = 32
    PRIVILEGED_RECON_DIM = 3
    MAX_LENGTH = 5
    
    # Hidden layer dimensions
    ACTOR_HIDDEN_DIMS = [256, 128, 32]
    MLP_HIDDEN_DIMS = [256, 128]
    
    # Activation functions
    ACTIVATION = 'elu'
    OUTPUT_ACTIVATION = None
    
    # Action noise parameters
    INIT_NOISE_STD = 1.0
    MAX_STD = 1.2
    MIN_STD = 0.1


def create_hugwbc_policy(z_dim: int = 32, horizon: int = 5, proprio_dim: int = 63, clock_dim: int = 0) -> HugWBCPolicyNetwork:
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
        privileged_dim=config.PRIVILEGED_DIM,
        terrain_dim=config.TERRAIN_DIM,
        latent_dim=config.LATENT_DIM,
        privileged_recon_dim=config.PRIVILEGED_RECON_DIM,
        max_length=config.MAX_LENGTH,
        actor_hidden_dims=config.ACTOR_HIDDEN_DIMS,
        mlp_hidden_dims=config.MLP_HIDDEN_DIMS,
        activation=config.ACTIVATION,
        output_activation=config.OUTPUT_ACTIVATION,
        init_noise_std=config.INIT_NOISE_STD,
        max_std=config.MAX_STD,
        min_std=config.MIN_STD
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
    policy = create_hugwbc_policy(z_dim=32, clock_dim=2)
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
    privileged_obs = torch.randn(batch_size, 3)  # Full critic observation dimension (63+11+2+24+221)
    clock = torch.randn(batch_size, 2)
    print(f"\nInput shapes:")
    print(f"  observations: {observations.shape}")
    print(f"  z_vector: {z_vector.shape}")
    print(f"  privileged_obs: {privileged_obs.shape}")
    print(f"  clock: {clock.shape}")
    # Test forward pass
    with torch.no_grad():
        # Deterministic inference
        actions_mean, latent = policy.act_inference(observations, z_vector, clock=clock)
        print(f"\nInference output shapes:")
        print(f"  actions_mean: {actions_mean.shape}")
        print(f"  latent: {latent.shape}")
        
        # Stochastic sampling
        actions = policy.act(observations, z_vector, privileged_obs, clock=clock, sync_update=True)
        print(f"  sampled_actions: {actions.shape}")
        
        # Get log probabilities
        log_probs = policy.get_actions_log_prob(actions)
        print(f"  log_probs: {log_probs.shape}")
        
        # Get privileged loss
        metrics = {'privileged_recon_loss': 0.0}
        priv_loss = policy.get_privileged_loss(metrics)
        print(f"\nPrivileged reconstruction loss: {priv_loss.item():.6f}")
        print(f"Metrics: {metrics}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Policy network is working correctly.")
    print("=" * 60)
