"""Masked actor and scalar critic for SubWeb-Frontier-A2C."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional


class MaskedCandidateActor(nn.Module):
    """Shared scorer over padded candidate action slots."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, context_dim: int = 0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.embedding_dim = 64
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
        )
        self.head = nn.Linear(self.embedding_dim, 1)

    def _features(
        self,
        action_mat: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = action_mat.float()
        if self.context_dim:
            if context is None:
                raise ValueError("masked actor requires coverage context")
            ctx = context.float().view(1, -1).expand(features.shape[0], -1)
            features = torch.cat([features, ctx], dim=-1)
        return features

    def action_embeddings(
        self,
        action_mat: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(self._features(action_mat, context))

    def forward(
        self,
        action_mat: torch.Tensor,
        action_mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.any(action_mask):
            raise ValueError("masked actor received no valid actions")
        logits = self.head(self.action_embeddings(action_mat, context)).squeeze(-1)
        return logits.masked_fill(~action_mask.bool(), -1.0e9)

    def get_action_dist(
        self,
        action_mat: torch.Tensor,
        action_mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Categorical:
        return Categorical(logits=self.forward(action_mat, action_mask, context))


class CandidateQHead(nn.Module):
    """Action-conditioned return head over the actor's candidate embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        recurrent_hidden_dim: int = 0,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim + self.recurrent_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = embeddings.float()
        if self.recurrent_hidden_dim:
            if hidden is None:
                raise ValueError("candidate Q head requires recurrent hidden state")
            h_vec = hidden.float().view(1, -1).expand(features.shape[0], -1)
            features = torch.cat([features, h_vec], dim=-1)
        values = self.net(features).squeeze(-1)
        return values.masked_fill(~action_mask.bool(), 0.0)


class FrontierCritic(nn.Module):
    """Scalar value model for future marginal frontier coverage."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.net(critic_input.float()).squeeze(-1)


def masked_mean(action_mat: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = action_mask.float().unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (action_mat.float() * mask).sum(dim=0) / denom
