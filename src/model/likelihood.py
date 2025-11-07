import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from src.utils.logger import Logger



logger = Logger(__name__)

# ---- Masking utility (for Union Gene Set) ----
def union_mask(batch_genes, union_gene_list):
    """Returns a binary mask for batch gene list against union gene set."""
    gene_to_idx = {g: i for i, g in enumerate(union_gene_list)}
    mask = torch.zeros(len(union_gene_list), dtype=torch.bool)
    for g in batch_genes:
        idx = gene_to_idx.get(g, None)
        if idx is not None:
            mask[idx] = 1
    return mask

# ---- ZINB Parameter Neural Networks ----
class MuDecoder(nn.Module):
    def __init__(self, latent_dim, n_batches, n_genes, n_hidden=128, n_layers=2):
        super().__init__()
        self.input_dim = latent_dim + n_batches
        self.n_genes = n_genes
        layers = [nn.Linear(self.input_dim, n_hidden), nn.ReLU()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(n_hidden, n_genes)
    def forward(self, z_bn, batch_onehot, lib_size=None):
        # z_bn: (batch, latent_dim), batch_onehot: (batch, n_batches), lib_size: (batch,) or None
        x = torch.cat([z_bn, batch_onehot], dim=1)
        h = self.mlp(x)
        mu = self.output(h)
        if lib_size is not None:
            mu = mu + lib_size.unsqueeze(1)    # broadcasting log(s_bn)
        return torch.exp(mu)    # mean in NB must be positive

class PiDecoder(nn.Module):
    def __init__(self, latent_dim, n_batches, n_genes, n_hidden=128, n_layers=2):
        super().__init__()
        self.input_dim = latent_dim + n_batches
        self.n_genes = n_genes
        layers = [nn.Linear(self.input_dim, n_hidden), nn.ReLU()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(n_hidden, n_genes)
    def forward(self, z_bn, batch_onehot):
        x = torch.cat([z_bn, batch_onehot], dim=1)
        h = self.mlp(x)
        pi = torch.sigmoid(self.output(h))  # dropout probability in (0,1)
        return pi

# ---- Gene-specific Dispersion ----
class DispersionParams(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.theta_g = nn.Parameter(torch.ones(n_genes))
    def forward(self):
        return F.softplus(self.theta_g)

# ---- ZINB Likelihood (with Masking for Measured Genes) ----
def zinb_likelihood(x, mu, theta, pi, mask):
    # x: (..., n_genes), mask: (..., n_genes) bool
    try:
        # Negative Binomial log-prob
        t1 = torch.lgamma(theta + x) - torch.lgamma(x+1) - torch.lgamma(theta)
        t2 = theta * (torch.log(theta) - torch.log(theta+mu))
        t3 = x * (torch.log(mu) - torch.log(theta+mu))
        nb_logp = t1 + t2 + t3  # (..., n_genes)
        zinb_logp = torch.log(pi + 1e-8) * (x == 0).float() + torch.log((1-pi) * torch.exp(nb_logp) + 1e-8) * (x != 0).float()
        masked = zinb_logp * mask.float()
        # Example reduction: sum loglikelihood over measured genes, mean over batch
        return masked.sum(-1).mean()
    except Exception as ex:
        logger.error(f"Failed in ZINB likelihood computation: {ex}")
        raise

# ---- Unified Likelihood Module (Supervised & Unsupervised) ----
class UnifiedLikelihood(nn.Module):
    def __init__(self, latent_dim, n_batches, union_gene_list, n_hidden=128, n_layers=2):
        super().__init__()
        self.n_batches = n_batches
        self.union_gene_list = union_gene_list
        self.n_genes = len(union_gene_list)
        self.mu_decoder = MuDecoder(latent_dim, n_batches, self.n_genes, n_hidden, n_layers)
        self.pi_decoder = PiDecoder(latent_dim, n_batches, self.n_genes, n_hidden, n_layers)
        self.theta = DispersionParams(self.n_genes)
    def forward(self, z_bn, batch_onehot, x_bn, lib_size, mask):
        mu = self.mu_decoder(z_bn, batch_onehot, lib_size)
        pi = self.pi_decoder(z_bn, batch_onehot)
        theta = self.theta()  # n_genes
        loss = -zinb_likelihood(x_bn, mu, theta, pi, mask)
        return loss, {'mu': mu, 'pi': pi, 'theta': theta}

# Usage example (model init/build, input/output shapes, masking):
# model = UnifiedLikelihood(latent_dim, n_batches, union_gene_list)
# loss, outs = model(z_bn, batch_onehot, x_bn, lib_size, mask)
#  z_bn: (batch, latent_dim), batch_onehot: (batch, n_batches), x_bn: (batch, n_genes), lib_size: (batch,)
#  mask: (batch, n_genes) bool -- 1 only on measured genes of batch

# Note: This file is intended to be placed in scARKIDS/src/model/likelihood.py
#       Ensure that you use logger for ALL key events, warnings, errors.
#       Further error handling and checks (dimensions, NaNs, etc) can be easily extended within each module/class.
#       Extend for supervised/unsupervised (cell-type priors) at higher-level dispatch if needed.
