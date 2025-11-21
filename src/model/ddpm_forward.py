"""
DDPM Forward Process Module for scARKIDS
=========================================

Implements the forward (noising) diffusion process for VAE-DDPM model.

Mathematical Background:
-----------------------

Forward Process:
  q(z^(1:T) | z^(0)) = ∏_{t=1}^{T} q(z^(t) | z^(t-1))

Single Step (t → t+1):
  q(z^(t) | z^(t-1)) = N(z^(t) | √(1 - β_t) z^(t-1), β_t I)
  
Closed-Form (direct sampling from z^(0) to z^(t)):
  q(z^(t) | z^(0)) = N(z^(t) | √(ᾱ_t) z^(0), (1 - ᾱ_t) I)

Where:
  - β_t: Variance schedule at timestep t
  - ᾱ_t: Cumulative product of (1 - β_t)
  - Key: This process is FIXED (no learnable parameters)

Note: Forward process is identical for supervised and unsupervised cases.
      Cell type information affects upstream (encoder) and downstream (reverse process).
"""

from src.utils.logger import Logger
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DDPMForwardConfig:
    """
    Configuration for the DDPM Forward Process module.
    
    Attributes:
        latent_dim: Dimension of latent space z^(0)
        n_diffusion_steps: Number of diffusion timesteps T
        beta_schedule: Type of variance schedule ('linear', 'cosine', 'quadratic')
        beta_min: Minimum β_t value
        beta_max: Maximum β_t value
    """
    latent_dim: int
    n_diffusion_steps: int
    beta_schedule: str = 'linear'
    beta_min: float = 1e-4
    beta_max: float = 2e-2
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.n_diffusion_steps > 0, "n_diffusion_steps must be positive"
        assert self.beta_schedule in ['linear', 'cosine', 'quadratic'], \
            f"beta_schedule must be one of ['linear', 'cosine', 'quadratic'], got {self.beta_schedule}"
        assert 0 < self.beta_min < self.beta_max < 1, \
            f"Must have 0 < beta_min < beta_max < 1, got beta_min={self.beta_min}, beta_max={self.beta_max}"

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Variance Schedule Computation
# ============================================================================

def compute_beta_schedule(
    n_steps: int,
    schedule_type: str,
    beta_min: float,
    beta_max: float
) -> torch.Tensor:
    """
    Compute variance schedule β_t.
    
    Args:
        n_steps: Number of diffusion steps T
        schedule_type: Type of schedule ('linear', 'cosine', 'quadratic')
        beta_min: Minimum β value
        beta_max: Maximum β value
        
    Returns:
        Tensor of shape (n_steps,) containing β_t values
    """
    if schedule_type == 'linear':
        # Linear interpolation: β_t = β_min + (t/T) * (β_max - β_min)
        beta = torch.linspace(beta_min, beta_max, n_steps)
        logger.debug(f"Using linear β schedule")
        
    elif schedule_type == 'cosine':
        # Cosine schedule from Nichol & Dhariwal 2021
        s = 0.008
        steps = torch.arange(n_steps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        beta = torch.clip(betas, 0.0001, 0.9999)
        logger.debug(f"Using cosine β schedule")
        
    elif schedule_type == 'quadratic':
        # Quadratic schedule: β_t = (√β_min + (t/T)(√β_max - √β_min))²
        beta = torch.linspace(np.sqrt(beta_min), np.sqrt(beta_max), n_steps) ** 2
        logger.debug(f"Using quadratic β schedule")
        
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
        
    return beta

def compute_variance_schedule(config: DDPMForwardConfig) -> Dict[str, torch.Tensor]:
    """
    Compute all variance schedule components.
    
    Mathematical relationships:
      α_t = 1 - β_t
      ᾱ_t = ∏_{i=1}^{t} α_i
      
    Args:
        config: Configuration object
        
    Returns:
        Dictionary containing all schedule tensors
    """
    # Compute β_t
    beta = compute_beta_schedule(
        config.n_diffusion_steps,
        config.beta_schedule,
        config.beta_min,
        config.beta_max
    )
    
    # Compute α_t = 1 - β_t
    alpha = 1.0 - beta
    
    # Compute ᾱ_t = ∏ α_i
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    
    # Compute ᾱ_{t-1} (shifted version, prepend 1.0 for t=0)
    alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)
    
    # Pre-compute square roots for efficiency
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    sqrt_beta = torch.sqrt(beta)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)
    
    logger.info(
        f"Variance schedule computed: β ∈ [{beta[0]:.6f}, {beta[-1]:.6f}], "
        f"ᾱ ∈ [{alpha_cumprod[-1]:.6f}, {alpha_cumprod[0]:.6f}]"
    )
    
    return {
        'beta': beta,
        'alpha': alpha,
        'alpha_cumprod': alpha_cumprod,
        'alpha_cumprod_prev': alpha_cumprod_prev,
        'sqrt_alpha_cumprod': sqrt_alpha_cumprod,
        'sqrt_one_minus_alpha_cumprod': sqrt_one_minus_alpha_cumprod,
        'sqrt_beta': sqrt_beta,
        'sqrt_one_minus_alpha': sqrt_one_minus_alpha,
    }

# ============================================================================
# DDPM Forward Process Module
# ============================================================================

class DDPMForwardModule(nn.Module):
    """
    Forward diffusion process module.
    
    Implements noise addition using pre-computed variance schedules.
    This process is FIXED (no learnable parameters).
    """
    
    def __init__(self, config: DDPMForwardConfig):
        """
        Initialize forward process module.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Compute variance schedules (pre-computed, not trainable)
        schedules = compute_variance_schedule(config)
        
        # Register as buffers (not parameters, moves with model to device)
        for key, value in schedules.items():
            self.register_buffer(key, value)
        
        logger.info("Initialized DDPMForwardModule")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  n_diffusion_steps={config.n_diffusion_steps}")
        logger.info(f"  beta_schedule={config.beta_schedule}")
    
    def _validate_inputs(self, z: torch.Tensor, t: torch.Tensor) -> None:
        """
        Validate input tensors.
        
        Args:
            z: Latent tensor
            t: Timestep tensor
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check shapes
        if z.dim() != 2:
            raise ValueError(f"z must be 2D (batch_size, latent_dim), got shape {z.shape}")
        if z.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch. Expected {self.config.latent_dim}, got {z.shape[-1]}"
            )
        if t.dim() != 1:
            raise ValueError(f"t must be 1D (batch_size,), got shape {t.shape}")
        if z.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch size mismatch. z: {z.shape[0]}, t: {t.shape[0]}"
            )
        
        # Check timestep range
        if (t < 0).any() or (t >= self.config.n_diffusion_steps).any():
            raise ValueError(
                f"Timestep out of range [0, {self.config.n_diffusion_steps-1}]. "
                f"Got min={t.min()}, max={t.max()}"
            )
        
        # Check for NaN/Inf
        if torch.isnan(z).any() or torch.isinf(z).any():
            raise ValueError("z contains NaN or Inf values")
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("t contains NaN or Inf values")
    
    def add_noise_single_step(
        self,
        z_t_minus_1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise for a single diffusion step (t-1 → t).
        
        Mathematical formulation:
          q(z^(t) | z^(t-1)) = N(√(1-β_t) z^(t-1), β_t I)
          
        Implementation:
          z^(t) = √(1-β_t) * z^(t-1) + √β_t * ε,  ε ~ N(0, I)
        
        Args:
            z_t_minus_1: Latent at step t-1, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), values in [0, T-1]
            
        Returns:
            z_t: Noised latent at step t, shape (batch_size, latent_dim)
            noise: Sampled Gaussian noise, shape (batch_size, latent_dim)
        """
        self._validate_inputs(z_t_minus_1, t)
        
        # Sample Gaussian noise: ε ~ N(0, I)
        noise = torch.randn_like(z_t_minus_1)
        
        # Index schedule: √(1-β_t) and √β_t
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha[t].reshape(-1, 1)
        sqrt_beta_t = self.sqrt_beta[t].reshape(-1, 1)
        
        # Compute: z^(t) = √(1-β_t) * z^(t-1) + √β_t * ε
        z_t = sqrt_one_minus_alpha_t * z_t_minus_1 + sqrt_beta_t * noise
        
        return z_t, noise
    
    def add_noise_closed_form(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise directly from z^(0) to z^(t) using closed-form.
        
        Mathematical formulation:
          q(z^(t) | z^(0)) = N(√ᾱ_t z^(0), (1-ᾱ_t) I)
          
        Implementation:
          z^(t) = √ᾱ_t * z^(0) + √(1-ᾱ_t) * ε,  ε ~ N(0, I)
        
        This is the PRIMARY method used during training (more efficient than iterating).
        
        Args:
            z_0: Initial VAE latent, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), values in [0, T-1]
            
        Returns:
            z_t: Noised latent at step t, shape (batch_size, latent_dim)
            noise: Sampled Gaussian noise, shape (batch_size, latent_dim)
        """
        self._validate_inputs(z_0, t)
        
        # Sample Gaussian noise: ε ~ N(0, I)
        noise = torch.randn_like(z_0)
        
        # Index schedule: √ᾱ_t and √(1-ᾱ_t)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1)
        
        # Compute: z^(t) = √ᾱ_t * z^(0) + √(1-ᾱ_t) * ε
        z_t = sqrt_alpha_cumprod_t * z_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return z_t, noise
    
    def sample_random_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps uniformly for a batch.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Timesteps in [0, T-1], shape (batch_size,)
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        t = torch.randint(
            0,
            self.config.n_diffusion_steps,
            (batch_size,),
            device=self.beta.device
        )
        return t
    
    def get_schedule_at_timestep(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get variance schedule components at specific timesteps.
        
        Args:
            t: Timestep indices, shape (batch_size,)
            
        Returns:
            Dictionary of schedule components at timesteps t
        """
        if (t < 0).any() or (t >= self.config.n_diffusion_steps).any():
            raise ValueError(
                f"Timestep out of range [0, {self.config.n_diffusion_steps-1}]"
            )
        
        return {
            'beta': self.beta[t],
            'alpha': self.alpha[t],
            'alpha_cumprod': self.alpha_cumprod[t],
            'sqrt_alpha_cumprod': self.sqrt_alpha_cumprod[t],
            'sqrt_one_minus_alpha_cumprod': self.sqrt_one_minus_alpha_cumprod[t],
            'sqrt_beta': self.sqrt_beta[t],
            'sqrt_one_minus_alpha': self.sqrt_one_minus_alpha[t],
        }
    
    def forward(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: add noise using closed-form.
        
        Args:
            z_0: Initial VAE latent, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
            
        Returns:
            z_t: Noised latent, shape (batch_size, latent_dim)
            noise: Sampled noise, shape (batch_size, latent_dim)
        """
        return self.add_noise_closed_form(z_0, t)

# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class DDPMForwardManager:
    """
    Manager class for the DDPM Forward Process module.
    
    This is the single entry point that:
    1. Parses configuration from config.yaml
    2. Initializes the forward process module
    3. Exposes APIs for training/inference
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize manager from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing 'ddpm_forward' section from config.yaml
        """
        logger.info("Initializing DDPMForwardManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Initialize forward process module
        self.forward_module = DDPMForwardModule(self.config)
        
        logger.info("DDPMForwardManager initialized successfully")
    
    def _parse_config(self, config_dict: Dict) -> DDPMForwardConfig:
        """
        Parse configuration dictionary into DDPMForwardConfig.
        
        Args:
            config_dict: Dictionary from config.yaml
            
        Returns:
            DDPMForwardConfig object
        """
        try:
            config = DDPMForwardConfig(
                latent_dim=config_dict['latent_dim'],
                n_diffusion_steps=config_dict['n_diffusion_steps'],
                beta_schedule=config_dict.get('beta_schedule', 'linear'),
                beta_min=config_dict.get('beta_min', 1e-4),
                beta_max=config_dict.get('beta_max', 2e-2)
            )
            logger.info("Configuration parsed successfully")
            return config
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def get_module(self) -> DDPMForwardModule:
        """
        Get the forward process module.
        
        Returns:
            DDPMForwardModule instance
        """
        return self.forward_module
    
    def add_noise(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise to latent codes (main API for training/inference).
        
        Args:
            z_0: Initial VAE latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,). If None, sample randomly
            
        Returns:
            z_t: Noised latent, shape (batch_size, latent_dim)
            noise: Sampled noise, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
        """
        # Sample timesteps if not provided
        if t is None:
            t = self.forward_module.sample_random_timesteps(z_0.shape[0])
        
        # Add noise using closed-form
        z_t, noise = self.forward_module.add_noise_closed_form(z_0, t)
        
        return z_t, noise, t

    def add_noise_single_step(
        self,
        z_t_minus_1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise for a single diffusion step (t-1 → t).
        
        Args:
            z_t_minus_1: Latent at step t-1, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
        
        Returns:
            z_t: Noised latent at step t, shape (batch_size, latent_dim)
            noise: Sampled Gaussian noise, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
        """
        z_t, noise = self.forward_module.add_noise_single_step(z_t_minus_1, t)
        return z_t, noise, t

    def get_schedule(self, t: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Get variance schedule components.
        
        Args:
            t: Specific timesteps. If None, return full schedule
            
        Returns:
            Dictionary of schedule tensors
        """
        if t is None:
            # Return full schedule
            return {
                'beta': self.forward_module.beta,
                'alpha': self.forward_module.alpha,
                'alpha_cumprod': self.forward_module.alpha_cumprod,
                'alpha_cumprod_prev': self.forward_module.alpha_cumprod_prev,
                'sqrt_alpha_cumprod': self.forward_module.sqrt_alpha_cumprod,
                'sqrt_one_minus_alpha_cumprod': self.forward_module.sqrt_one_minus_alpha_cumprod,
                'sqrt_beta': self.forward_module.sqrt_beta,
                'sqrt_one_minus_alpha': self.forward_module.sqrt_one_minus_alpha,
            }
        else:
            # Return schedule at specific timesteps
            return self.forward_module.get_schedule_at_timestep(t)
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters (empty, as forward process is fixed)
        """
        return dict(self.forward_module.named_parameters())

# ============================================================================
# Config YAML Schema Documentation
# ============================================================================

"""
Example config.yaml section for ddpm_forward:
----------------------------------------------

ddpm_forward:
  # Required parameters
  latent_dim: 10                    # Dimension of latent space z^(0)
  n_diffusion_steps: 1000           # Number of diffusion timesteps T
  
  # Optional parameters (with defaults)
  beta_schedule: 'linear'           # Variance schedule type: 'linear', 'cosine', 'quadratic'
  beta_min: 1.0e-4                  # Minimum β_t value
  beta_max: 2.0e-2                  # Maximum β_t value


Example usage:
--------------

```python
import yaml
from ddpm_forward import DDPMForwardManager

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize manager
manager = DDPMForwardManager(config['ddpm_forward'])

# Get module for training
forward_module = manager.get_module()

# Add noise during training
z_0 = encoder(x_bn)  # From VAE encoder
z_t, noise, t = manager.add_noise(z_0)

# Get schedule components (for reverse process)
schedule = manager.get_schedule()
schedule_at_t = manager.get_schedule(t)
```
"""
