"""
DDPM Backward (Reverse) Process Module for scARKIDS
====================================================

Implements the reverse (denoising) diffusion process for VAE-DDPM model.

Mathematical Background:
-----------------------

Reverse Process (Supervised - known cell type c*):
    p_ψ(z^(0:T) | c*) = p(z^(T)) ∏_{t=1}^{T} p_ψ(z^(t-1) | z^(t), c*)

Reverse Process (Unsupervised - predicted cell type c):
    p_ψ(z^(0:T) | c) = p(z^(T)) ∏_{t=1}^{T} p_ψ(z^(t-1) | z^(t), c)

Single Reverse Step:
    p_ψ(z^(t-1) | z^(t), c) = N(z^(t-1) | μ_ψ(z^(t), t, c), Σ_ψ(z^(t), t, c))

Parameterization (noise prediction formulation):
    μ_ψ(z^(t), t, c) = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε_ψ(z^(t), t, c))

where ε_ψ is a neural network predicting noise.

Variance:
    Σ_ψ = σ_t² I, where σ_t² = β_t (fixed) or learned

Key Difference from Forward:
    - Forward process is FIXED (no learnable parameters)
    - Reverse process is LEARNED (ε_ψ network is trained)

Note: The same reverse network handles both supervised and unsupervised modes.
The only difference is whether c is known (supervised) or predicted (unsupervised).
"""

from src.utils.logger import Logger
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DDPMBackwardConfig:
    """
    Configuration for the DDPM Backward Process module.
    
    Attributes:
        latent_dim: Dimension of latent space z^(0)
        n_diffusion_steps: Number of diffusion timesteps T
        n_cell_types: Number of cell types C
        variance_type: Variance strategy ('fixed' or 'learned')
        noise_hidden_dim: Hidden dimension for noise prediction network
        noise_n_layers: Number of layers in noise prediction network
        timestep_embed_dim: Dimension of timestep embedding
        celltype_embed_dim: Dimension of cell type embedding
        dropout: Dropout rate for noise prediction network
    """
    
    latent_dim: int
    n_diffusion_steps: int
    n_cell_types: int
    variance_type: str = 'fixed'
    noise_hidden_dim: int = 128
    noise_n_layers: int = 2
    timestep_embed_dim: int = 64
    celltype_embed_dim: int = 32
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.n_diffusion_steps > 0, "n_diffusion_steps must be positive"
        assert self.n_cell_types > 0, "n_cell_types must be positive"
        assert self.variance_type in ['fixed', 'learned'], \
            f"variance_type must be 'fixed' or 'learned', got {self.variance_type}"
        assert self.noise_hidden_dim > 0, "noise_hidden_dim must be positive"
        assert self.noise_n_layers > 0, "noise_n_layers must be positive"
        assert self.timestep_embed_dim > 0, "timestep_embed_dim must be positive"
        assert self.celltype_embed_dim > 0, "celltype_embed_dim must be positive"
        assert 0 <= self.dropout < 1, f"dropout must be in [0, 1), got {self.dropout}"

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Sinusoidal Timestep Embedding
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    
    Converts discrete timestep indices t ∈ [0, T-1] to continuous embeddings
    using sinusoidal functions at different frequencies.
    
    Formula:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    
    def __init__(self, embed_dim: int):
        """
        Initialize sinusoidal timestep embedding.
        
        Args:
            embed_dim: Dimension of timestep embedding (must be even)
        """
        super().__init__()
        
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        
        self.embed_dim = embed_dim
        
        # Pre-compute frequency factors
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embeddings for timesteps.
        
        Args:
            t: Timestep indices, shape (batch_size,), dtype long
        
        Returns:
            Timestep embeddings, shape (batch_size, embed_dim)
        """
        if len(t.shape) != 1:
            raise ValueError(f"t must be 1D tensor, got shape {t.shape}")
        
        # t: (batch_size,) -> (batch_size, 1)
        t = t.float().unsqueeze(-1)
        
        # Compute arguments: t * freqs -> (batch_size, half_dim)
        args = t * self.freqs.unsqueeze(0)
        
        # Compute sin and cos embeddings
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embeddings  # (batch_size, embed_dim)

# ============================================================================
# Noise Prediction Network
# ============================================================================

class NoisePredictionNetwork(nn.Module):
    """
    Neural network ε_ψ(z^(t), t, c) that predicts noise added at timestep t.
    
    Architecture:
        1. Embed timestep t using sinusoidal encoding
        2. Embed cell type c using learned embedding
        3. Concatenate: [z^(t), timestep_embed, celltype_embed]
        4. Process through MLP
        5. Output: predicted noise ε̂ (same shape as z^(t))
    
    The network is conditioned on both timestep and cell type.
    """
    
    def __init__(self, config: DDPMBackwardConfig):
        """
        Initialize noise prediction network.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
        """
        super().__init__()
        
        self.config = config
        
        # Timestep embedding: t -> embedding
        self.timestep_embedding = SinusoidalTimestepEmbedding(config.timestep_embed_dim)
        
        # Cell type embedding: c -> embedding (learnable)
        self.celltype_embedding = nn.Embedding(
            num_embeddings=config.n_cell_types,
            embedding_dim=config.celltype_embed_dim
        )
        
        # Input dimension: latent + timestep_embed + celltype_embed
        input_dim = config.latent_dim + config.timestep_embed_dim + config.celltype_embed_dim
        
        # Build MLP
        layers = []
        current_dim = input_dim
        
        for i in range(config.noise_n_layers):
            layers.append(nn.Linear(current_dim, config.noise_hidden_dim))
            layers.append(nn.SiLU())  # Smooth activation for diffusion models
            layers.append(nn.Dropout(config.dropout))
            current_dim = config.noise_hidden_dim
        
        # Output layer: predict noise (same dimension as latent)
        layers.append(nn.Linear(current_dim, config.latent_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized NoisePredictionNetwork: input_dim={input_dim}, "
                   f"hidden_dim={config.noise_hidden_dim}, n_layers={config.noise_n_layers}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise ε̂ = ε_ψ(z^(t), t, c).
        
        Args:
            z_t: Noisy latent at timestep t, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,), dtype long
            cell_type: Cell type indices, shape (batch_size,), dtype long
        
        Returns:
            Predicted noise ε̂, shape (batch_size, latent_dim)
        """
        # Embed timestep: (batch_size,) -> (batch_size, timestep_embed_dim)
        t_embed = self.timestep_embedding(t)
        
        # Embed cell type: (batch_size,) -> (batch_size, celltype_embed_dim)
        c_embed = self.celltype_embedding(cell_type)
        
        # Concatenate all inputs: (batch_size, total_input_dim)
        x = torch.cat([z_t, t_embed, c_embed], dim=-1)
        
        # Predict noise through MLP: (batch_size, latent_dim)
        noise_pred = self.mlp(x)
        
        return noise_pred

# ============================================================================
# Variance Predictor (for learned variance)
# ============================================================================

class VariancePredictor(nn.Module):
    """
    Optional learned variance predictor.
    
    Predicts log variance log(σ_t²) for the reverse process distribution.
    Can improve sample quality compared to fixed variance.
    """
    
    def __init__(self, config: DDPMBackwardConfig):
        """
        Initialize variance predictor.
        
        Args:
            config: DDPMBackwardConfig with hyperparameters
        """
        super().__init__()
        
        self.config = config
        
        # Reuse same embedding modules
        self.timestep_embedding = SinusoidalTimestepEmbedding(config.timestep_embed_dim)
        self.celltype_embedding = nn.Embedding(
            num_embeddings=config.n_cell_types,
            embedding_dim=config.celltype_embed_dim
        )
        
        # Input dimension
        input_dim = config.latent_dim + config.timestep_embed_dim + config.celltype_embed_dim
        
        # Smaller MLP for variance prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.noise_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.noise_hidden_dim // 2, config.latent_dim)
        )
        
        logger.info("Initialized VariancePredictor for learned variance")
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict log variance log(σ²).
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Log variance, shape (batch_size, latent_dim)
        """
        # Embed inputs
        t_embed = self.timestep_embedding(t)
        c_embed = self.celltype_embedding(cell_type)
        
        # Concatenate and predict
        x = torch.cat([z_t, t_embed, c_embed], dim=-1)
        log_variance = self.mlp(x)
        
        return log_variance

# ============================================================================
# DDPM Backward Process Module
# ============================================================================

class DDPMBackwardModule(nn.Module):
    """
    Reverse diffusion process module.
    
    Implements denoising using learned noise prediction network.
    This process is LEARNED (trainable parameters in ε_ψ network).
    """
    
    def __init__(
        self,
        config: DDPMBackwardConfig,
        variance_schedule: Dict[str, torch.Tensor]
    ):
        """
        Initialize backward process module.
        
        Args:
            config: Configuration object
            variance_schedule: Variance schedule from forward process
                Required keys: beta, alpha, alpha_cumprod, sqrt_alpha_cumprod,
                             sqrt_one_minus_alpha_cumprod
        """
        super().__init__()
        
        self.config = config
        
        # Register variance schedule as buffers (not trainable)
        for key, value in variance_schedule.items():
            self.register_buffer(key, value)
        
        # Validate variance schedule
        self._validate_variance_schedule()
        
        # Initialize noise prediction network (learnable)
        self.noise_network = NoisePredictionNetwork(config)
        
        # Initialize variance predictor (if learned variance)
        if config.variance_type == 'learned':
            self.variance_predictor = VariancePredictor(config)
            logger.info("Using learned variance")
        else:
            self.variance_predictor = None
            logger.info("Using fixed variance (β_t)")
        
        logger.info("Initialized DDPMBackwardModule")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  n_diffusion_steps={config.n_diffusion_steps}")
        logger.info(f"  n_cell_types={config.n_cell_types}")
        logger.info(f"  variance_type={config.variance_type}")
    
    def _validate_variance_schedule(self):
        """Validate that all required variance schedule components are present."""
        required_keys = [
            'beta', 'alpha', 'alpha_cumprod',
            'sqrt_alpha_cumprod', 'sqrt_one_minus_alpha_cumprod'
        ]
        
        for key in required_keys:
            if not hasattr(self, key):
                raise ValueError(f"Missing variance schedule component: {key}")
            
            schedule_tensor = getattr(self, key)
            if schedule_tensor.shape[0] != self.config.n_diffusion_steps:
                raise ValueError(
                    f"Variance schedule {key} has wrong length: "
                    f"expected {self.config.n_diffusion_steps}, got {schedule_tensor.shape[0]}"
                )
    
    def _validate_inputs(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> None:
        """
        Validate input tensors.
        
        Args:
            z_t: Noisy latent tensor
            t: Timestep tensor
            cell_type: Cell type tensor
        
        Raises:
            ValueError: If inputs are invalid
        """
        batch_size = z_t.shape[0]
        
        # Check shapes
        if z_t.dim() != 2:
            raise ValueError(f"z_t must be 2D (batch_size, latent_dim), got shape {z_t.shape}")
        
        if z_t.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch. Expected {self.config.latent_dim}, got {z_t.shape[-1]}"
            )
        
        if t.dim() != 1:
            raise ValueError(f"t must be 1D (batch_size,), got shape {t.shape}")
        
        if t.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch. z_t: {batch_size}, t: {t.shape[0]}")
        
        if cell_type.dim() != 1:
            raise ValueError(f"cell_type must be 1D (batch_size,), got shape {cell_type.shape}")
        
        if cell_type.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch. z_t: {batch_size}, cell_type: {cell_type.shape[0]}")
        
        # Check timestep range
        if (t < 0).any() or (t >= self.config.n_diffusion_steps).any():
            raise ValueError(
                f"Timestep out of range [0, {self.config.n_diffusion_steps-1}]. "
                f"Got min={t.min()}, max={t.max()}"
            )
        
        # Check cell type range
        if (cell_type < 0).any() or (cell_type >= self.config.n_cell_types).any():
            raise ValueError(
                f"Cell type out of range [0, {self.config.n_cell_types-1}]. "
                f"Got min={cell_type.min()}, max={cell_type.max()}"
            )
        
        # Check for NaN/Inf
        if torch.isnan(z_t).any() or torch.isinf(z_t).any():
            raise ValueError("z_t contains NaN or Inf values")
        
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("t contains NaN or Inf values")
        
        if torch.isnan(cell_type).any() or torch.isinf(cell_type).any():
            raise ValueError("cell_type contains NaN or Inf values")
    
    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise ε̂ = ε_ψ(z^(t), t, c).
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Predicted noise, shape (batch_size, latent_dim)
        """
        self._validate_inputs(z_t, t, cell_type)
        return self.noise_network(z_t, t, cell_type)
    
    def compute_mean(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor,
        noise_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean μ_ψ(z^(t), t, c) using noise prediction formulation.
        
        Mathematical formula:
            μ_ψ(z^(t), t, c) = (1/√α_t) * (z^(t) - (β_t/√(1-ᾱ_t)) * ε̂)
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
            noise_pred: Optional pre-computed noise prediction
        
        Returns:
            Mean μ, shape (batch_size, latent_dim)
        """
        # Predict noise if not provided
        if noise_pred is None:
            noise_pred = self.predict_noise(z_t, t, cell_type)
        
        # Extract schedule components at timestep t
        beta_t = self.beta[t].reshape(-1, 1)  # (batch_size, 1)
        alpha_t = self.alpha[t].reshape(-1, 1)  # (batch_size, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1)
        
        # Compute coefficient: β_t / √(1 - ᾱ_t)
        coef = beta_t / sqrt_one_minus_alpha_cumprod_t
        
        # Compute mean: μ = (1/√α_t) * (z^(t) - coef * ε̂)
        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - coef * noise_pred)
        
        return mean
    
    def compute_variance(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variance σ² of reverse process distribution.
        
        Two options:
            1. Fixed: σ_t² = β_t (standard DDPM)
            2. Learned: σ_t² = exp(log_var_pred) (improved DDPM)
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Variance σ², shape (batch_size, latent_dim) or (batch_size, 1)
        """
        if self.config.variance_type == 'fixed':
            # Fixed variance: σ_t² = β_t
            beta_t = self.beta[t]  # (batch_size,)
            variance = beta_t.reshape(-1, 1)  # (batch_size, 1) for broadcasting
        
        elif self.config.variance_type == 'learned':
            # Learned variance: σ_t² = exp(log_var)
            log_variance = self.variance_predictor(z_t, t, cell_type)
            variance = torch.exp(log_variance)  # (batch_size, latent_dim)
        
        else:
            raise ValueError(f"Unknown variance_type: {self.config.variance_type}")
        
        return variance
    
    def reverse_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform single reverse step: sample z^(t-1) ~ p_ψ(z^(t-1) | z^(t), c).
        
        Mathematical formulation:
            z^(t-1) = μ_ψ(z^(t), t, c) + σ_t * ε', where ε' ~ N(0, I)
        
        Special case: at t=0, no noise is added (deterministic).
        
        Args:
            z_t: Noisy latent at timestep t, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
            cell_type: Cell type indices, shape (batch_size,)
        
        Returns:
            z_t_minus_1: Denoised latent at timestep t-1, shape (batch_size, latent_dim)
            info_dict: Dictionary with intermediate values
        """
        self._validate_inputs(z_t, t, cell_type)
        
        # Compute mean
        mean = self.compute_mean(z_t, t, cell_type)
        
        # Compute variance
        variance = self.compute_variance(z_t, t, cell_type)
        
        # Sample noise: ε' ~ N(0, I)
        noise = torch.randn_like(z_t)
        
        # Compute standard deviation: σ = √(variance)
        std = torch.sqrt(variance)
        
        # Sample: z^(t-1) = μ + σ * ε'
        # Special case: at t=0, no noise is added (deterministic)
        no_noise = (t == 0).float().reshape(-1, 1)  # (batch_size, 1)
        z_t_minus_1 = mean + (1.0 - no_noise) * std * noise
        
        # Return results and info
        info_dict = {
            'mean': mean,
            'variance': variance,
            'std': std,
            'noise_added': noise
        }
        
        return z_t_minus_1, info_dict
    
    def sample_trajectory(
        self,
        n_samples: int,
        cell_type: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Sample full trajectory: z^(T) → z^(T-1) → ... → z^(0).
        
        Sampling process:
            1. Start from Gaussian noise: z^(T) ~ N(0, I)
            2. For t = T, T-1, ..., 1:
                  z^(t-1) ~ p_ψ(z^(t-1) | z^(t), c)
            3. Return z^(0)
        
        Args:
            n_samples: Number of samples to generate
            cell_type: Cell type for each sample, shape (n_samples,), dtype long
            return_trajectory: If True, return full trajectory of latents
        
        Returns:
            z_0: Final denoised latent, shape (n_samples, latent_dim)
            trajectory: Optional list of latents at each timestep (if return_trajectory=True)
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        if cell_type.shape[0] != n_samples:
            raise ValueError(
                f"cell_type batch size mismatch: expected {n_samples}, got {cell_type.shape[0]}"
            )
        
        # Initialize trajectory storage (if requested)
        trajectory = [] if return_trajectory else None
        
        # Start from Gaussian noise: z^(T) ~ N(0, I)
        z_t = torch.randn(n_samples, self.config.latent_dim, device=self.beta.device)
        
        if return_trajectory:
            trajectory.append(z_t.clone())
        
        logger.debug(f"Sampling: Starting from z^(T) ~ N(0, I), shape={z_t.shape}")
        
        # Iteratively denoise: t = T-1, T-2, ..., 0
        for step_idx in range(self.config.n_diffusion_steps - 1, -1, -1):
            # Create timestep tensor: (n_samples,)
            t = torch.full((n_samples,), step_idx, dtype=torch.long, device=self.beta.device)
            
            # Perform reverse step: z^(t) → z^(t-1)
            z_t, info_dict = self.reverse_step(z_t, t, cell_type)
            
            # Store trajectory
            if return_trajectory:
                trajectory.append(z_t.clone())
            
            # Log progress periodically
            if (step_idx + 1) % 100 == 0 or step_idx == 0:
                logger.debug(
                    f"Sampling: Step {self.config.n_diffusion_steps - step_idx}/"
                    f"{self.config.n_diffusion_steps}, t={step_idx}, "
                    f"z_norm={z_t.norm(dim=1).mean():.4f}"
                )
        
        logger.info(
            f"Sampling complete: Generated {n_samples} samples, "
            f"final z^(0) norm={z_t.norm(dim=1).mean():.4f}"
        )
        
        return z_t, trajectory
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: perform single reverse step.
        
        Args:
            z_t: Noisy latent at timestep t, shape (batch_size, latent_dim)
            t: Timestep indices, shape (batch_size,)
            cell_type: Cell type indices, shape (batch_size,)
        
        Returns:
            z_t_minus_1: Denoised latent, shape (batch_size, latent_dim)
            info_dict: Dictionary with intermediate values
        """
        return self.reverse_step(z_t, t, cell_type)

# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class DDPMBackwardManager:
    """
    Manager class for the DDPM Backward Process module.
    
    This is the single entry point that:
        1. Parses configuration from config.yaml
        2. Initializes the backward process module
        3. Exposes APIs for training/inference
    """
    
    def __init__(
        self,
        config_dict: Dict,
        variance_schedule: Dict[str, torch.Tensor]
    ):
        """
        Initialize manager from configuration dictionary and variance schedule.
        
        Args:
            config_dict: Dictionary containing 'ddpm_backward' section from config.yaml
            variance_schedule: Variance schedule from DDPMForwardManager
        """
        logger.info("Initializing DDPMBackwardManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Store variance schedule
        self.variance_schedule = variance_schedule
        
        # Initialize backward process module
        self.backward_module = DDPMBackwardModule(self.config, variance_schedule)
        
        logger.info("DDPMBackwardManager initialized successfully")
    
    def _parse_config(self, config_dict: Dict) -> DDPMBackwardConfig:
        """
        Parse configuration dictionary into DDPMBackwardConfig.
        
        Args:
            config_dict: Dictionary from config.yaml
        
        Returns:
            DDPMBackwardConfig object
        """
        try:
            config = DDPMBackwardConfig(
                latent_dim=config_dict['latent_dim'],
                n_diffusion_steps=config_dict['n_diffusion_steps'],
                n_cell_types=config_dict['n_cell_types'],
                variance_type=config_dict.get('variance_type', 'fixed'),
                noise_hidden_dim=config_dict.get('noise_hidden_dim', 128),
                noise_n_layers=config_dict.get('noise_n_layers', 2),
                timestep_embed_dim=config_dict.get('timestep_embed_dim', 64),
                celltype_embed_dim=config_dict.get('celltype_embed_dim', 32),
                dropout=config_dict.get('dropout', 0.1)
            )
            
            logger.info("Configuration parsed successfully")
            return config
        
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def get_module(self) -> DDPMBackwardModule:
        """
        Get the backward process module.
        
        Returns:
            DDPMBackwardModule instance
        """
        return self.backward_module
    
    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise ε̂ = ε_ψ(z^(t), t, c).
        
        This is the main API used during training to compute the denoising loss.
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            Predicted noise, shape (batch_size, latent_dim)
        """
        return self.backward_module.predict_noise(z_t, t, cell_type)
    
    def reverse_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform single reverse step: z^(t) → z^(t-1).
        
        Args:
            z_t: Noisy latent, shape (batch_size, latent_dim)
            t: Timesteps, shape (batch_size,)
            cell_type: Cell types, shape (batch_size,)
        
        Returns:
            z_t_minus_1: Denoised latent, shape (batch_size, latent_dim)
            info_dict: Dictionary with intermediate values
        """
        return self.backward_module.reverse_step(z_t, t, cell_type)
    
    def sample(
        self,
        n_samples: int,
        cell_type: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Sample from reverse process: z^(T) → ... → z^(0).
        
        This is the main API used during inference to generate new samples.
        
        Args:
            n_samples: Number of samples to generate
            cell_type: Cell type for each sample, shape (n_samples,)
            return_trajectory: If True, return full trajectory
        
        Returns:
            z_0: Final denoised latent, shape (n_samples, latent_dim)
            trajectory: Optional list of latents (if return_trajectory=True)
        """
        return self.backward_module.sample_trajectory(n_samples, cell_type, return_trajectory)
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters (includes noise network and optional variance predictor)
        """
        return dict(self.backward_module.named_parameters())

# ============================================================================
# Config YAML Schema Documentation
# ============================================================================

"""
Example usage:
--------------

```python
import yaml
from ddpm_forward import DDPMForwardManager
from ddpm_backward import DDPMBackwardManager

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize forward process (to get variance schedule)
forward_manager = DDPMForwardManager(config['ddpm_forward'])
variance_schedule = forward_manager.get_schedule()

# Initialize backward process
backward_manager = DDPMBackwardManager(
    config['ddpm_backward'],
    variance_schedule
)

# Get module for training
backward_module = backward_manager.get_module()

# Training: predict noise for loss computation
z_t, noise_true, t = forward_manager.add_noise(z_0)
noise_pred = backward_manager.predict_noise(z_t, t, cell_type)
loss = F.mse_loss(noise_pred, noise_true)

# Inference: sample new latents
cell_type = torch.tensor([0, 1, 2, 3, 4] * 10)  # 50 samples, mixed cell types
z_0_generated = backward_manager.sample(n_samples=50, cell_type=cell_type)
```
"""
