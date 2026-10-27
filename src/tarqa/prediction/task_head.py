from __future__ import annotations

import torch
from torch import nn


class QueryConditionedTaskHead(nn.Module):
    """Question-conditioned fusion of deterministic relational features and LLM embeddings."""

    def __init__(
        self,
        relational_input_dim: int,
        question_input_dim: int,
        hidden_dim: int = 256,
        question_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.relational_encoder = nn.Sequential(
            nn.Linear(relational_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.question_projection = nn.Sequential(
            nn.Linear(question_input_dim, question_dim),
            nn.LayerNorm(question_dim),
            nn.GELU(),
        )
        self.film = nn.Linear(question_dim, hidden_dim * 2)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + question_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, relational_features: torch.Tensor, question_features: torch.Tensor) -> torch.Tensor:
        relation_hidden = self.relational_encoder(relational_features)
        question_hidden = self.question_projection(question_features)
        gamma, beta = self.film(question_hidden).chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        modulated = gamma * relation_hidden + beta
        return self.output(torch.cat([modulated, question_hidden], dim=-1)).squeeze(-1)
