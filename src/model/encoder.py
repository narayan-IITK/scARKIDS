"""
Encoder Module for scARKIDS
============================

Implements VAE encoder for inferring initial latent representation z^(0).

Mathematical Background:
-----------------------

VAE Encoder (Supervised Setting):
q_φ(z^(0) | x, b, c*) = N(z^(0) | μ_φ(x, b, c*), diag(σ²_φ(x, b, c*)))

VAE Encoder (Unsupervised Setting):
q_φ(z^(0) | x, b, c) = N(z^(0) | μ_φ(x, b, c), diag(σ²_φ(x, b, c)))

Reparameterization Trick:
z^(0) = μ_φ(x, b, c*) + σ_φ(x, b, c*) ⊙ ε,    ε ~ N(0, I)

Where:
- x: Gene expression vector (n_genes,)
- b: Batch indicator (one-hot, n_batches)
- c or c*: Cell type indicator (one-hot, n_cell_types)
- μ_φ: Mean prediction network
- σ²_φ: Variance prediction network (diagonal covariance)
- z^(0): Initial latent representation (latent_dim,)

Input Encoding:
log(x + 1) normalization applied to gene expression counts
"""

from src.utils.logger import Logger
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EncoderConfig:
    """
    Configuration for the Encoder module.
    
    Attributes:
        n_genes: Number of input genes
        n_batches: Number of batches
        n_cell_types: Number of cell types
        latent_dim: Dimension of latent space z^(0)
        hidden_dims: List of hidden layer dimensions for encoder MLP
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        use_layer_norm: Whether to use layer normalization (alternative to batch norm)
        input_transform: Input transformation type: 'log1p' or 'none'
        eps: Small constant for numerical stability
    """
    n_genes: int
    n_batches: int
    n_cell_types: int
    latent_dim: int
    hidden_dims: list = None  # Default set in __post_init__
    dropout: float = 0.1
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    input_transform: str = 'log1p'
    eps: float = 1e-8
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.n_genes > 0, "n_genes must be positive"
        assert self.n_batches > 0, "n_batches must be positive"
        assert self.n_cell_types > 0, "n_cell_types must be positive"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert self.input_transform in ['log1p', 'none'], \
            "input_transform must be 'log1p' or 'none'"
        assert self.eps > 0, "eps must be positive"
        assert not (self.use_batch_norm and self.use_layer_norm), \
            "Cannot use both batch_norm and layer_norm simultaneously"
        
        # Set default hidden dimensions if not provided
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)


# ============================================================================
# Encoder Neural Network
# ============================================================================

class EncoderMLP(nn.Module):
    """
    Multi-layer perceptron for encoding input to latent space.
    
    Architecture:
        Input: [x_transformed, batch_onehot, celltype_onehot] 
        → Hidden layers (with normalization, activation, dropout)
        → Two output heads (μ and log(σ²))
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        dropout: float,
        use_batch_norm: bool,
        use_layer_norm: bool
    ):
        """
        Args:
            input_dim: Input dimension (n_genes + n_batches + n_cell_types)
            output_dim: Output dimension (latent_dim)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear transformation
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads for mean and log-variance
        self.fc_mean = nn.Linear(current_dim, output_dim)
        self.fc_logvar = nn.Linear(current_dim, output_dim)
        
        logger.debug(
            f"Initialized EncoderMLP: {input_dim} → "
            f"{hidden_dims} → {output_dim} (μ and log σ²)"
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            mean: Mean μ_φ of shape (batch_size, output_dim)
            logvar: Log-variance log(σ²_φ) of shape (batch_size, output_dim)
        """
        # Shared encoder
        h = self.encoder(x)
        
        # Separate heads for mean and log-variance
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        
        return mean, logvar


# ============================================================================
# VAE Encoder Module
# ============================================================================

class VAEEncoder(nn.Module):
    """
    VAE Encoder Module implementing q_φ(z^(0) | x, b, c).
    
    Produces Gaussian approximate posterior over latent space:
    q_φ(z^(0) | x, b, c) = N(z^(0) | μ_φ(x, b, c), diag(σ²_φ(x, b, c)))
    
    Supports both supervised (c* known) and unsupervised (c inferred) settings.
    """
    
    def __init__(self, config: EncoderConfig):
        """
        Initialize VAE encoder.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Compute input dimension
        # Input: [gene_expression, batch_onehot, celltype_onehot]
        input_dim = config.n_genes + config.n_batches + config.n_cell_types
        
        # Encoder MLP
        self.encoder_mlp = EncoderMLP(
            input_dim=input_dim,
            output_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm
        )
        
        logger.info("Initialized VAEEncoder")
        logger.info(f"  n_genes={config.n_genes}")
        logger.info(f"  n_batches={config.n_batches}")
        logger.info(f"  n_cell_types={config.n_cell_types}")
        logger.info(f"  latent_dim={config.latent_dim}")
        logger.info(f"  hidden_dims={config.hidden_dims}")
        logger.info(f"  input_transform={config.input_transform}")
    
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply input transformation to gene expression counts.
        
        Args:
            x: Raw gene expression counts (batch_size, n_genes)
        
        Returns:
            Transformed expression (batch_size, n_genes)
        """
        if self.config.input_transform == 'log1p':
            # log(x + 1) transformation - standard for scRNA-seq
            return torch.log1p(x)
        elif self.config.input_transform == 'none':
            return x
        else:
            raise ValueError(f"Unknown input_transform: {self.config.input_transform}")
    
    def encode(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters (μ, σ²).
        
        Args:
            x: Gene expression counts (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
        
        Returns:
            mean: μ_φ(x, b, c) of shape (batch_size, latent_dim)
            logvar: log(σ²_φ(x, b, c)) of shape (batch_size, latent_dim)
        """
        # Transform input
        x_transformed = self._transform_input(x)
        
        # Concatenate all inputs
        encoder_input = torch.cat([x_transformed, batch_onehot, celltype_onehot], dim=1)
        
        # Encode to latent parameters
        mean, logvar = self.encoder_mlp(encoder_input)
        
        return mean, logvar
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for sampling from N(μ, σ²).
        
        z^(0) = μ + σ ⊙ ε,  where ε ~ N(0, I)
        
        Args:
            mean: Mean μ (batch_size, latent_dim)
            logvar: Log-variance log(σ²) (batch_size, latent_dim)
        
        Returns:
            z: Sampled latent variable z^(0) (batch_size, latent_dim)
        """
        # Compute standard deviation: σ = exp(0.5 * log(σ²)) = exp(log σ)
        std = torch.exp(0.5 * logvar)
        
        # Sample ε ~ N(0, I)
        eps = torch.randn_like(std)
        
        # Reparameterization: z = μ + σ ⊙ ε
        z = mean + std * eps
        
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: torch.Tensor,
        return_params: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass: encode input and sample latent variable.
        
        Args:
            x: Gene expression counts (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
            return_params: If True, return (z, (μ, log σ²)); else return z only
        
        Returns:
            z: Sampled latent variable z^(0) (batch_size, latent_dim)
            params: Optional tuple (mean, logvar) if return_params=True
        """
        # Encode to latent parameters
        mean, logvar = self.encode(x, batch_onehot, celltype_onehot)
        
        # Sample latent variable using reparameterization trick
        z = self.reparameterize(mean, logvar)
        
        if return_params:
            return z, (mean, logvar)
        else:
            return z
    
    #TODO: Decide whether to keep this function here or in elbo.py, my guess it move it to elbo.py


# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class EncoderManager:
    """
    Manager class for the Encoder module.
    
    This is the single entry point that:
    1. Parses configuration from config.yaml
    2. Initializes the encoder module
    3. Exposes APIs for training/inference
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize manager from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing 'encoder' section from config.yaml
        """
        logger.info("Initializing EncoderManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Initialize encoder module
        self.encoder_module = VAEEncoder(self.config)
        
        logger.info("EncoderManager initialized successfully")
    
    def _parse_config(self, config_dict: Dict) -> EncoderConfig:
        """
        Parse configuration dictionary into EncoderConfig.
        
        Args:
            config_dict: Dictionary from config.yaml
        
        Returns:
            EncoderConfig object
        """
        try:
            config = EncoderConfig(
                n_genes=config_dict['n_genes'],
                n_batches=config_dict['n_batches'],
                n_cell_types=config_dict['n_cell_types'],
                latent_dim=config_dict['latent_dim'],
                hidden_dims=config_dict.get('hidden_dims', None),
                dropout=config_dict.get('dropout', 0.1),
                use_batch_norm=config_dict.get('use_batch_norm', True),
                use_layer_norm=config_dict.get('use_layer_norm', False),
                input_transform=config_dict.get('input_transform', 'log1p'),
                eps=config_dict.get('eps', 1e-8)
            )
            
            logger.info("Configuration parsed successfully")
            return config
            
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def get_module(self) -> VAEEncoder:
        """
        Get the encoder module.
        
        Returns:
            VAEEncoder instance
        """
        return self.encoder_module
    
    def encode(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        celltype_onehot: torch.Tensor,
        sample: bool = True,
        return_params: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Encode input to latent representation (main API for training/inference).
        
        Args:
            x: Gene expression counts (batch_size, n_genes)
            batch_onehot: Batch indicator (batch_size, n_batches)
            celltype_onehot: Cell type indicator (batch_size, n_cell_types)
            sample: If True, sample z using reparameterization; else return mean
            return_params: If True, also return (mean, logvar)
        
        Returns:
            z: Latent representation z^(0) (batch_size, latent_dim)
            params: Optional tuple (mean, logvar) if return_params=True
        """
        # Encode to latent parameters
        mean, logvar = self.encoder_module.encode(x, batch_onehot, celltype_onehot)
        
        # Sample or use mean
        if sample:
            z = self.encoder_module.reparameterize(mean, logvar)
        else:
            z = mean  # Use mean for deterministic encoding (e.g., inference)
        
        if return_params:
            return z, (mean, logvar)
        else:
            return z
    
    #TODO: Decide whether to expose an API to compute kl div between encoder posterior and prior here or in elbo.py, my guess is elbo.py
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters
        """
        return dict(self.encoder_module.named_parameters())



