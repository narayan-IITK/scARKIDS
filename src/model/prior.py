"""
scARKIDS Prior Module

Implements prior distributions for VAE-DDPM models (supervised and unsupervised).
Supports union gene set masking for variable gene dimensions across batches.

Mathematical Framework:
- Supervised: p(z^(0)|c*) = N(z^(0)|μ_c, Σ_c)
- Unsupervised: p(z^(0)) = N(z^(0)|0, I)
- Terminal Diffusion: p(z^(T)) = N(z^(T)|0, I)
- Cell Type Prior: p(c) = Categorical(c|π_0)
- Batch Prior: p(b) = Categorical(b|ρ)
"""

from src.utils.logger import Logger
from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn



logger = Logger.get_logger(__name__)



# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PriorConfig:
    """Configuration for prior distributions with validation."""
    latent_dim: int
    n_cell_types: int
    n_batches: int
    supervised: bool
    device: str = "cpu"

    def __post_init__(self):
        """Validate all parameters."""
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        if self.n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be > 0, got {self.n_cell_types}")
        if self.n_batches <= 0:
            raise ValueError(f"n_batches must be > 0, got {self.n_batches}")
        
        logger.info(
            f"PriorConfig: latent_dim={self.latent_dim}, "
            f"n_cell_types={self.n_cell_types}, n_batches={self.n_batches}, "
            f"supervised={self.supervised}"
        )


# ============================================================================
# ABSTRACT BASE PRIOR
# ============================================================================

class Prior(ABC, nn.Module):
    """Abstract base class for all prior distributions."""

    def __init__(self, config: PriorConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def log_prob(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log probability. Args: z (batch_size, dim). Returns: (batch_size,)"""
        pass

    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Sample from prior. Returns: (n_samples, dim)"""
        pass


# ============================================================================
# VAE LATENT PRIORS
# ============================================================================

class VAELatentPriorSupervised(Prior):
    """
    Supervised VAE latent prior: p(z^(0)|c*) = N(z^(0)|μ_c, Σ_c)
    
    Cell-type-specific means and variances (learnable).
    Different cell types occupy different regions of latent space.
    """

    def __init__(self, config: PriorConfig):
        super().__init__(config)
        if not config.supervised:
            raise ValueError("VAELatentPriorSupervised requires supervised=True")

        # Learnable cell-type-specific parameters
        self.register_parameter(
            "means",
            nn.Parameter(torch.randn(config.n_cell_types, config.latent_dim))
        )
        self.register_parameter(
            "log_vars",
            nn.Parameter(torch.zeros(config.n_cell_types, config.latent_dim))
        )
        logger.info(f"Initialized VAELatentPriorSupervised ({config.n_cell_types} cell types)")

    def log_prob(self, z: torch.Tensor, cell_type: torch.Tensor) -> torch.Tensor:
        """Compute log p(z^(0)|c*) for cell-type-specific Gaussian."""
        if z.shape[0] != cell_type.shape[0]:
            raise ValueError(f"Batch size mismatch: z={z.shape[0]}, c={cell_type.shape[0]}")

        means = self.means[cell_type]
        log_vars = self.log_vars[cell_type]
        vars = torch.exp(log_vars)

        # log N(z|μ,σ²) = -0.5 * [(z-μ)²/σ² + log(σ²) + log(2π)]
        diff = z - means
        log_prob = -0.5 * (
            torch.sum(diff ** 2 / vars, dim=1) +
            torch.sum(log_vars, dim=1) +
            self.config.latent_dim * np.log(2 * np.pi)
        )
        return log_prob

    def sample(self, n_samples: int, cell_type: torch.Tensor) -> torch.Tensor:
        """Sample from p(z^(0)|c*) using reparameterization trick."""
        if cell_type.shape[0] != n_samples:
            raise ValueError(f"n_samples mismatch: {n_samples} vs {cell_type.shape[0]}")

        means = self.means[cell_type]
        stds = torch.exp(0.5 * self.log_vars[cell_type])
        epsilon = torch.randn_like(means)
        return means + stds * epsilon


class VAELatentPriorUnsupervised(Prior):
    """
    Unsupervised VAE latent prior: p(z^(0)) = N(z^(0)|0, I)
    
    Shared standard Gaussian (no learnable parameters).
    Cell types are inferred through a separate classifier.
    """

    def __init__(self, config: PriorConfig):
        super().__init__(config)
        if config.supervised:
            raise ValueError("VAELatentPriorUnsupervised requires supervised=False")
        logger.info(f"Initialized VAELatentPriorUnsupervised (standard Gaussian)")

    def log_prob(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log p(z^(0)) = log N(z^(0)|0, I)."""
        if z.shape[-1] != self.config.latent_dim:
            raise ValueError(f"Dimension mismatch: {z.shape[-1]} vs {self.config.latent_dim}")

        # log N(z|0,I) = -0.5 * [||z||² + d*log(2π)]
        log_prob = -0.5 * (
            torch.sum(z ** 2, dim=1) +
            self.config.latent_dim * np.log(2 * np.pi)
        )
        return log_prob

    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Sample from N(0, I)."""
        return torch.randn(n_samples, self.config.latent_dim, device=self.device)


# ============================================================================
# CATEGORICAL PRIORS
# ============================================================================

class CellTypePrior(Prior):
    """
    Cell type prior: p(c) = Categorical(c|π_0)
    
    Typically uniform: π_0 = (1/C, ..., 1/C) for C cell types.
    Can be initialized from empirical frequencies.
    """

    def __init__(
        self,
        config: PriorConfig,
        probabilities: Optional[torch.Tensor] = None
    ):
        super().__init__(config)

        if probabilities is None:
            probabilities = torch.ones(config.n_cell_types) / config.n_cell_types
            logger.info("CellTypePrior: uniform distribution")
        else:
            if probabilities.shape[0] != config.n_cell_types:
                raise ValueError(f"Shape mismatch: {probabilities.shape[0]} vs {config.n_cell_types}")
            if not torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6):
                raise ValueError("probabilities must sum to 1")
            logger.info("CellTypePrior: empirical frequencies")

        self.register_buffer("probabilities", probabilities.to(self.device))

    def log_prob(self, c: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log p(c)."""
        if c.dtype not in [torch.int64, torch.long]:
            raise ValueError(f"cell_type must be dtype long, got {c.dtype}")
        return torch.log(self.probabilities[c])

    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Sample from Categorical(π_0)."""
        samples = torch.multinomial(
            self.probabilities.unsqueeze(0).expand(n_samples, -1),
            num_samples=1
        ).squeeze(1)
        return samples

    def update_from_empirical(self, cell_types: torch.Tensor) -> None:
        """Update probabilities from observed cell type frequencies."""
        counts = torch.bincount(
            cell_types.long(),
            minlength=self.config.n_cell_types
        ).float()
        self.probabilities = counts / counts.sum()
        logger.info("Updated CellTypePrior from empirical frequencies")


class BatchPrior(Prior):
    """
    Batch prior: p(b) = Categorical(b|ρ)
    
    Typically uniform or empirical frequencies.
    Integrates with union gene set masking strategy.
    """

    def __init__(
        self,
        config: PriorConfig,
        probabilities: Optional[torch.Tensor] = None
    ):
        super().__init__(config)

        if probabilities is None:
            probabilities = torch.ones(config.n_batches) / config.n_batches
            logger.info("BatchPrior: uniform distribution")
        else:
            if probabilities.shape[0] != config.n_batches:
                raise ValueError(f"Shape mismatch: {probabilities.shape[0]} vs {config.n_batches}")
            if not torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6):
                raise ValueError("probabilities must sum to 1")
            logger.info("BatchPrior: empirical frequencies")

        self.register_buffer("probabilities", probabilities.to(self.device))

    def log_prob(self, b: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log p(b)."""
        if b.dtype not in [torch.int64, torch.long]:
            raise ValueError(f"batch must be dtype long, got {b.dtype}")
        return torch.log(self.probabilities[b])

    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Sample from Categorical(ρ)."""
        samples = torch.multinomial(
            self.probabilities.unsqueeze(0).expand(n_samples, -1),
            num_samples=1
        ).squeeze(1)
        return samples


# ============================================================================
# TERMINAL DIFFUSION PRIOR
# ============================================================================

class TerminalDiffusionPrior(Prior):
    """
    Terminal diffusion prior: p(z^(T)) = N(z^(T)|0, I)
    
    Standard Gaussian at final diffusion step (fixed, no learnable parameters).
    Same for both supervised and unsupervised cases.
    """

    def __init__(self, config: PriorConfig):
        super().__init__(config)
        logger.info("Initialized TerminalDiffusionPrior (standard Gaussian)")

    def log_prob(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log p(z^(T)) = log N(z^(T)|0, I)."""
        if z.shape[-1] != self.config.latent_dim:
            raise ValueError(f"Dimension mismatch: {z.shape[-1]} vs {self.config.latent_dim}")

        log_prob = -0.5 * (
            torch.sum(z ** 2, dim=1) +
            self.config.latent_dim * np.log(2 * np.pi)
        )
        return log_prob

    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Sample from N(0, I)."""
        return torch.randn(n_samples, self.config.latent_dim, device=self.device)


# ============================================================================
# UNIFIED PRIOR MANAGER
# ============================================================================

class PriorManager(nn.Module):
    """
    Unified manager for all prior distributions.
    
    Composes individual priors, provides unified interface for log probability
    and sampling. Supports union gene set masking for variable gene dimensions.
    """

    def __init__(
        self,
        config: PriorConfig,
        cell_type_probabilities: Optional[torch.Tensor] = None,
        batch_probabilities: Optional[torch.Tensor] = None,
        union_gene_ids: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """
        Initialize prior manager.
        
        Args:
            config: PriorConfig with hyperparameters
            cell_type_probabilities: Shape (n_cell_types,), sums to 1
            batch_probabilities: Shape (n_batches,), sums to 1
            union_gene_ids: Dict mapping batch_idx to gene indices
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Initialize component priors
        if config.supervised:
            self.vae_latent_prior = VAELatentPriorSupervised(config)
        else:
            self.vae_latent_prior = VAELatentPriorUnsupervised(config)

        self.cell_type_prior = CellTypePrior(config, cell_type_probabilities)
        self.terminal_diffusion_prior = TerminalDiffusionPrior(config)
        self.batch_prior = BatchPrior(config, batch_probabilities)

        # Store gene masking info
        self.union_gene_ids = union_gene_ids or {}
        if union_gene_ids:
            logger.info(f"Union gene set masking: {len(union_gene_ids)} batches")

    def log_prob_vae_latent(
        self,
        z: torch.Tensor,
        cell_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log p(z^(0)|c*) or log p(z^(0)) depending on mode."""
        if self.config.supervised:
            if cell_type is None:
                raise ValueError("cell_type required for supervised prior")
            return self.vae_latent_prior.log_prob(z, cell_type=cell_type)
        return self.vae_latent_prior.log_prob(z)

    def log_prob_cell_type(self, c: torch.Tensor) -> torch.Tensor:
        """Compute log p(c)."""
        return self.cell_type_prior.log_prob(c)

    def log_prob_terminal_diffusion(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z^(T))."""
        return self.terminal_diffusion_prior.log_prob(z)

    def log_prob_batch(self, b: torch.Tensor) -> torch.Tensor:
        """Compute log p(b)."""
        return self.batch_prior.log_prob(b)

    def get_gene_mask(self, batch_idx: int, full_dim: int) -> torch.Tensor:
        """
        Get binary mask for genes measured in a batch.
        
        Returns all-ones if no masking info available.
        Shape: (full_dim,)
        """
        if batch_idx not in self.union_gene_ids:
            return torch.ones(full_dim, device=self.device, dtype=torch.float32)

        mask = torch.zeros(full_dim, device=self.device, dtype=torch.float32)
        measured_indices = self.union_gene_ids[batch_idx]
        mask[measured_indices] = 1.0
        return mask

    def sample_joint_prior(
        self,
        n_samples: int,
        cell_type: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample jointly from all priors.
        
        Returns:
            Dict with keys: "z_0" (VAE latent), "c" (cell type),
                           "b" (batch), "z_T" (terminal diffusion)
        """
        if cell_type is None:
            cell_type = self.cell_type_prior.sample(n_samples)
        if batch_idx is None:
            batch_idx = self.batch_prior.sample(n_samples)

        if self.config.supervised:
            z_0 = self.vae_latent_prior.sample(n_samples, cell_type=cell_type)
        else:
            z_0 = self.vae_latent_prior.sample(n_samples)

        z_T = self.terminal_diffusion_prior.sample(n_samples)

        return {
            "z_0": z_0,
            "c": cell_type,
            "b": batch_idx,
            "z_T": z_T,
        }

    def log_info(self) -> None:
        """Log detailed configuration information."""
        mode = "SUPERVISED" if self.config.supervised else "UNSUPERVISED"
        logger.info(f"PriorManager: mode={mode}, latent_dim={self.config.latent_dim}, "
                    f"n_cell_types={self.config.n_cell_types}, n_batches={self.config.n_batches}")
        logger.info(f"VAE Prior: {'Cell-type-specific Gaussian' if self.config.supervised else 'Standard Gaussian'}")
        logger.info(f"Cell Type Probs: {self.cell_type_prior.probabilities.cpu().numpy()}")
        logger.info(f"Batch Probs: {self.batch_prior.probabilities.cpu().numpy()}")
