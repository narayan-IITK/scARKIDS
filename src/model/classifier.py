"""
Cell Type Classifier Module for scARKIDS
========================================

Implements the cell type classifier that predicts cell type from gene expression.

Mathematical Background:
-----------------------
Cell Type Classifier (learnable, parameters ω):
    q_ω(c|x, b) = Categorical(c|π_ω(x, b))

where:
    - x: gene expression vector (n_genes,)
    - b: batch index
    - c: cell type (discrete, ∈ {0, 1, ..., C-1})
    - π_ω(x, b): neural network outputting class probabilities (C-dimensional simplex)

Architecture:
    1. Embed batch b using learned embedding
    2. Concatenate: [x, batch_embed]
    3. Process through MLP
    4. Output: logits → softmax → probabilities π_ω(x, b)

The classifier is used in two modes:
    - Training: Supervised (minimize cross-entropy with true labels c*)
    - Inference: Predict cell type from expression (argmax or sample from π_ω)
"""

from src.utils.logger import Logger
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ClassifierConfig:
    """
    Configuration for the Cell Type Classifier module.
    
    Attributes:
        n_genes: Number of genes in expression vector x
        n_cell_types: Number of cell types C
        n_batches: Number of batches B
        hidden_dim: Hidden layer dimension for MLP
        n_layers: Number of hidden layers in MLP
        batch_embed_dim: Dimension of batch embedding
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """
    n_genes: int
    n_cell_types: int
    n_batches: int
    hidden_dim: int = 256
    n_layers: int = 3
    batch_embed_dim: int = 32
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.n_genes > 0, "n_genes must be positive"
        assert self.n_cell_types > 0, "n_cell_types must be positive"
        assert self.n_batches > 0, "n_batches must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.batch_embed_dim > 0, "batch_embed_dim must be positive"
        assert 0 <= self.dropout < 1, f"dropout must be in [0, 1), got {self.dropout}"

# ============================================================================
# Logger
# ============================================================================

logger = Logger.get_logger(__name__)

# ============================================================================
# Classifier Network
# ============================================================================

class ClassifierNetwork(nn.Module):
    """
    Neural network π_ω(x, b) that maps (gene_expression, batch) to class probabilities.
    
    Architecture:
        Input: [x, batch_embed]
            x: gene expression, shape (n_genes,)
            batch_embed: batch embedding, shape (batch_embed_dim,)
        
        Hidden layers:
            Linear → BatchNorm (optional) → ReLU → Dropout
            (repeated n_layers times)
        
        Output layer:
            Linear → Softmax → class probabilities (n_cell_types,)
    
    Mathematical formulation:
        π_ω(x, b) = softmax(MLP([x, embed(b)]))
        
    where π_ω(x, b) ∈ Δ^(C-1) (probability simplex)
    """
    
    def __init__(self, config: ClassifierConfig):
        """
        Initialize classifier network.
        
        Args:
            config: ClassifierConfig with hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Batch embedding: b → embedding (learnable)
        self.batch_embedding = nn.Embedding(
            num_embeddings=config.n_batches,
            embedding_dim=config.batch_embed_dim
        )
        
        # Input dimension: gene expression + batch embedding
        input_dim = config.n_genes + config.batch_embed_dim
        
        # Build MLP layers
        layers = []
        current_dim = input_dim
        
        for i in range(config.n_layers):
            # Linear layer
            layers.append(nn.Linear(current_dim, config.hidden_dim))
            
            # Batch normalization (optional)
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(config.dropout))
            
            current_dim = config.hidden_dim
        
        # Output layer: hidden → logits (no activation, softmax applied later)
        layers.append(nn.Linear(current_dim, config.n_cell_types))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ClassifierNetwork: input_dim={input_dim}, "
                   f"hidden_dim={config.hidden_dim}, n_layers={config.n_layers}, "
                   f"n_cell_types={config.n_cell_types}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Compute class probabilities π_ω(x, b) or logits.
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,), dtype long
            return_logits: If True, return logits instead of probabilities
        
        Returns:
            If return_logits=False:
                Class probabilities π_ω(x, b), shape (batch_size, n_cell_types)
                Each row sums to 1 (probability simplex)
            If return_logits=True:
                Logits (unnormalized), shape (batch_size, n_cell_types)
        """
        # Embed batch: (batch_size,) → (batch_size, batch_embed_dim)
        batch_embed = self.batch_embedding(batch)
        
        # Concatenate gene expression and batch embedding
        # (batch_size, n_genes) + (batch_size, batch_embed_dim)
        # → (batch_size, n_genes + batch_embed_dim)
        features = torch.cat([x, batch_embed], dim=-1)
        
        # Pass through MLP to get logits
        logits = self.mlp(features)  # (batch_size, n_cell_types)
        
        if return_logits:
            return logits
        
        # Apply softmax to get probabilities (along class dimension)
        # π_ω(x, b) = softmax(logits)
        probs = F.softmax(logits, dim=-1)
        
        return probs

# ============================================================================
# Classifier Module
# ============================================================================

class ClassifierModule(nn.Module):
    """
    Cell type classifier module.
    
    Implements q_ω(c|x, b) = Categorical(c|π_ω(x, b))
    
    Functionality:
        - Training: Compute cross-entropy loss with true labels
        - Inference: Predict cell type (argmax or sample from distribution)
        - Evaluation: Compute accuracy and confidence metrics
    """
    
    def __init__(self, config: ClassifierConfig):
        """
        Initialize classifier module.
        
        Args:
            config: ClassifierConfig with hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Initialize classifier network
        self.network = ClassifierNetwork(config)
        
        logger.info("Initialized ClassifierModule")
        logger.info(f"  n_genes={config.n_genes}")
        logger.info(f"  n_cell_types={config.n_cell_types}")
        logger.info(f"  n_batches={config.n_batches}")
    
    def _validate_inputs(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        cell_type: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate input tensors.
        
        Args:
            x: Gene expression tensor
            batch: Batch index tensor
            cell_type: Optional cell type tensor (for training)
        
        Raises:
            ValueError: If inputs are invalid
        """
        batch_size = x.shape[0]
        
        # Check x shape
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (batch_size, n_genes), got shape {x.shape}")
        if x.shape[-1] != self.config.n_genes:
            raise ValueError(
                f"Gene dimension mismatch. Expected {self.config.n_genes}, got {x.shape[-1]}"
            )
        
        # Check batch shape
        if batch.dim() != 1:
            raise ValueError(f"batch must be 1D (batch_size,), got shape {batch.shape}")
        if batch.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch. x: {batch_size}, batch: {batch.shape[0]}")
        
        # Check batch range
        if (batch < 0).any() or (batch >= self.config.n_batches).any():
            raise ValueError(
                f"Batch index out of range [0, {self.config.n_batches-1}]. "
                f"Got min={batch.min()}, max={batch.max()}"
            )
        
        # Check cell_type if provided
        if cell_type is not None:
            if cell_type.dim() != 1:
                raise ValueError(f"cell_type must be 1D (batch_size,), got shape {cell_type.shape}")
            if cell_type.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch. x: {batch_size}, cell_type: {cell_type.shape[0]}"
                )
            if (cell_type < 0).any() or (cell_type >= self.config.n_cell_types).any():
                raise ValueError(
                    f"Cell type out of range [0, {self.config.n_cell_types-1}]. "
                    f"Got min={cell_type.min()}, max={cell_type.max()}"
                )
        
        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("x contains NaN or Inf values")
        if torch.isnan(batch).any() or torch.isinf(batch).any():
            raise ValueError("batch contains NaN or Inf values")
        if cell_type is not None and (torch.isnan(cell_type).any() or torch.isinf(cell_type).any()):
            raise ValueError("cell_type contains NaN or Inf values")
    
    def predict_probs(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class probabilities π_ω(x, b).
        
        Mathematical formulation:
            π_ω(x, b) = softmax(MLP([x, embed(b)]))
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
        
        Returns:
            Class probabilities, shape (batch_size, n_cell_types)
            Each row sums to 1
        """
        self._validate_inputs(x, batch)
        return self.network(x, batch, return_logits=False)
    
    def predict_class(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cell type class.
        
        Two modes:
            - Deterministic (sample=False): argmax of probabilities
                ĉ = argmax_c π_ω(x, b)_c
            
            - Stochastic (sample=True): sample from categorical distribution
                ĉ ~ Categorical(π_ω(x, b))
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
            sample: If True, sample from distribution; else use argmax
        
        Returns:
            predicted_class: Predicted cell type, shape (batch_size,)
            probs: Class probabilities, shape (batch_size, n_cell_types)
        """
        # Get probabilities
        probs = self.predict_probs(x, batch)
        
        if sample:
            # Sample from categorical distribution
            # ĉ ~ Categorical(π_ω(x, b))
            predicted_class = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Deterministic: argmax
            # ĉ = argmax_c π_ω(x, b)_c
            predicted_class = torch.argmax(probs, dim=-1)
        
        return predicted_class, probs
    
    def compute_loss(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        cell_type: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cross-entropy loss for supervised training.
        
        Mathematical formulation:
            L_classifier = -E_{(x,b,c*) ~ data}[log q_ω(c*|x, b)]
                        = -E[log π_ω(x, b)_{c*}]
        
        This is the negative log-likelihood of the true class under the
        predicted distribution.
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
            cell_type: True cell type labels, shape (batch_size,)
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            loss: Cross-entropy loss (scalar if reduction='mean'/'sum')
            info_dict: Dictionary with metrics
                - 'loss': loss value
                - 'accuracy': classification accuracy
                - 'confidence': mean predicted probability of true class
                - 'probs': predicted probabilities
        """
        self._validate_inputs(x, batch, cell_type)
        
        # Get logits (more numerically stable for cross-entropy)
        logits = self.network(x, batch, return_logits=True)
        
        # Compute cross-entropy loss
        # L = -log π_ω(x, b)_{c*}
        loss = F.cross_entropy(logits, cell_type, reduction=reduction)
        
        # Compute metrics (no gradient needed)
        with torch.no_grad():
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Accuracy: fraction of correct predictions
            predicted_class = torch.argmax(probs, dim=-1)
            accuracy = (predicted_class == cell_type).float().mean()
            
            # Confidence: mean probability assigned to true class
            # π_ω(x, b)_{c*}
            true_class_probs = probs[torch.arange(probs.shape[0]), cell_type]
            confidence = true_class_probs.mean()
        
        # Prepare info dictionary
        info_dict = {
            'loss': loss.item() if reduction != 'none' else loss.mean().item(),
            'accuracy': accuracy.item(),
            'confidence': confidence.item(),
            'probs': probs
        }
        
        return loss, info_dict
    
    def evaluate(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance on a batch.
        
        Computes:
            - Accuracy: fraction of correct predictions
            - Top-k accuracy: fraction where true class in top k predictions
            - Mean confidence: average probability of true class
            - Cross-entropy loss
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
            cell_type: True cell type labels, shape (batch_size,)
        
        Returns:
            Dictionary with evaluation metrics
        """
        self._validate_inputs(x, batch, cell_type)
        
        with torch.no_grad():
            # Get probabilities
            probs = self.predict_probs(x, batch)
            
            # Predicted class
            predicted_class = torch.argmax(probs, dim=-1)
            
            # Accuracy
            accuracy = (predicted_class == cell_type).float().mean().item()
            
            # Top-3 accuracy (if applicable)
            if self.config.n_cell_types >= 3:
                top3_pred = torch.topk(probs, k=min(3, self.config.n_cell_types), dim=-1).indices
                top3_accuracy = (top3_pred == cell_type.unsqueeze(-1)).any(dim=-1).float().mean().item()
            else:
                top3_accuracy = accuracy
            
            # Confidence (probability of true class)
            true_class_probs = probs[torch.arange(probs.shape[0]), cell_type]
            mean_confidence = true_class_probs.mean().item()
            
            # Cross-entropy loss
            logits = self.network(x, batch, return_logits=True)
            ce_loss = F.cross_entropy(logits, cell_type).item()
        
        metrics = {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'mean_confidence': mean_confidence,
            'cross_entropy': ce_loss
        }
        
        return metrics
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: predict class probabilities.
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
        
        Returns:
            Class probabilities π_ω(x, b), shape (batch_size, n_cell_types)
        """
        return self.predict_probs(x, batch)

# ============================================================================
# Manager Class (Module Entry Point)
# ============================================================================

class ClassifierManager:
    """
    Manager class for the Cell Type Classifier module.
    
    This is the single entry point that:
        1. Parses configuration from config.yaml
        2. Initializes the classifier module
        3. Exposes APIs for training/inference
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize manager from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing 'classifier' section from config.yaml
        """
        logger.info("Initializing ClassifierManager")
        
        # Parse configuration
        self.config = self._parse_config(config_dict)
        
        # Initialize classifier module
        self.classifier = ClassifierModule(self.config)
        
        logger.info("ClassifierManager initialized successfully")
    
    def _parse_config(self, config_dict: Dict) -> ClassifierConfig:
        """
        Parse configuration dictionary into ClassifierConfig.
        
        Args:
            config_dict: Dictionary from config.yaml
        
        Returns:
            ClassifierConfig object
        """
        try:
            config = ClassifierConfig(
                n_genes=config_dict['n_genes'],
                n_cell_types=config_dict['n_cell_types'],
                n_batches=config_dict['n_batches'],
                hidden_dim=config_dict.get('hidden_dim', 256),
                n_layers=config_dict.get('n_layers', 3),
                batch_embed_dim=config_dict.get('batch_embed_dim', 32),
                dropout=config_dict.get('dropout', 0.2),
                use_batch_norm=config_dict.get('use_batch_norm', True)
            )
            logger.info("Configuration parsed successfully")
            return config
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def get_module(self) -> ClassifierModule:
        """
        Get the classifier module.
        
        Returns:
            ClassifierModule instance
        """
        return self.classifier
    
    def predict(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cell type from gene expression.
        
        This is the main API used during inference.
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
            sample: If True, sample from distribution; else use argmax
        
        Returns:
            predicted_class: Predicted cell type, shape (batch_size,)
            probs: Class probabilities, shape (batch_size, n_cell_types)
        """
        return self.classifier.predict_class(x, batch, sample=sample)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        cell_type: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute classification loss.
        
        This is the main API used during training.
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
            cell_type: True cell type labels, shape (batch_size,)
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            loss: Cross-entropy loss
            info_dict: Dictionary with metrics
        """
        return self.classifier.compute_loss(x, batch, cell_type, reduction=reduction)
    
    def evaluate(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        cell_type: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            x: Gene expression, shape (batch_size, n_genes)
            batch: Batch indices, shape (batch_size,)
            cell_type: True cell type labels, shape (batch_size,)
        
        Returns:
            Dictionary with evaluation metrics
        """
        return self.classifier.evaluate(x, batch, cell_type)
    
    def get_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Dictionary of named parameters
        """
        return dict(self.classifier.named_parameters())
    


