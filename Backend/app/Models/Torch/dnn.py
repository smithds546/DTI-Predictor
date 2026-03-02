"""
Deep Neural Network for Drug-Target Interaction (DTI) prediction.

Architecture: dual-branch encoder (separate drug / protein sub-networks)
followed by a joint MLP interaction predictor.

  Drug  (167-dim MACCS)  ──► drug_encoder  ──► 128-dim
                                                         ►  concat (256-dim)
  Protein (1024-dim ProtBERT) ► prot_encoder ──► 128-dim
                                                         ►  predictor ──► logit

Design choices (vs. ReLu.py baseline):
  - Separate encoders let each modality learn its own representation.
  - Batch normalisation stabilises training and acts as implicit regularisation.
  - ReLU replaces sigmoid in hidden layers (avoids vanishing gradients in depth).
  - Dropout provides explicit regularisation.
  - BCEWithLogitsLoss is numerically safer and better-calibrated than MSE for
    binary classification.
  - Adam with weight-decay replaces manual SGD.
"""

import torch
import torch.nn as nn


class DrugEncoder(nn.Module):
    """Two-layer MLP for MACCS fingerprint input (167-dim)."""

    def __init__(self, input_dim: int = 167, hidden_dim: int = 256, out_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinEncoder(nn.Module):
    """Three-layer MLP for ProtBERT embedding input (1024-dim)."""

    def __init__(self, input_dim: int = 1024, hidden_dims: tuple = (512, 256),
                 out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),      # slightly less dropout in deeper layers
            nn.Linear(h2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractionPredictor(nn.Module):
    """Three-layer MLP that maps the fused drug+protein representation to a scalar logit.

    Deliberately leaner than a 4-layer version: fewer parameters reduces overfitting
    on small datasets (~12k samples) while still being expressive enough.
    """

    def __init__(self, in_dim: int = 256, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),               # raw logit (no sigmoid — used with BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DTI_DNN(nn.Module):
    """
    Full Drug-Target Interaction Deep Neural Network.

    Forward pass returns a raw scalar logit per sample (shape: [N, 1]).
    Use BCEWithLogitsLoss during training; apply torch.sigmoid for probabilities.

    Args:
        drug_input_dim  : dimensionality of drug features (default 167 for MACCS)
        prot_input_dim  : dimensionality of protein features (default 1024 for ProtBERT)
        drug_hidden     : hidden width in the drug encoder first layer
        prot_hiddens    : (h1, h2) widths in the protein encoder
        encoder_out     : shared output width for both encoders (concat size = 2x this)
        predictor_drop  : base dropout rate for the interaction predictor
        encoder_drop    : dropout rate inside the encoders
    """

    def __init__(
        self,
        drug_input_dim: int = 167,
        prot_input_dim: int = 1024,
        drug_hidden: int = 256,
        prot_hiddens: tuple = (512, 256),
        encoder_out: int = 128,
        predictor_drop: float = 0.4,
        encoder_drop: float = 0.3,
    ):
        super().__init__()
        self.drug_encoder = DrugEncoder(drug_input_dim, drug_hidden, encoder_out, encoder_drop)
        self.prot_encoder = ProteinEncoder(prot_input_dim, prot_hiddens, encoder_out, encoder_drop)
        self.predictor = InteractionPredictor(encoder_out * 2, predictor_drop)

    def forward(self, x_drug: torch.Tensor, x_prot: torch.Tensor) -> torch.Tensor:
        drug_repr = self.drug_encoder(x_drug)        # [N, 128]
        prot_repr = self.prot_encoder(x_prot)        # [N, 128]
        fused = torch.cat([drug_repr, prot_repr], dim=1)  # [N, 256]
        return self.predictor(fused)                 # [N, 1]

    def predict_proba(self, x_drug: torch.Tensor, x_prot: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities (useful at inference time)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x_drug, x_prot))