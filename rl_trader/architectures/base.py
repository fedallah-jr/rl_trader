"""Contract every actor-critic in this codebase must satisfy.

Any new architecture should subclass `AbstractActorCritic` and implement the
three methods below. Algorithms (`algorithms/*.py`) never reference a concrete
class — they consume this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractActorCritic(nn.Module, ABC):
    """Minimal observation → (action distribution, state value) interface.

    Conventions:
      - All tensors are batched; first dim = `B`.
      - Market has shape  [B, T, N_symbols, F_features].
      - Account has shape [B, N_symbols, F_account].
      - Globals has shape [B, F_globals].
      - Logits have shape [B, N_symbols, N_actions]   (factored per-symbol).
      - Value has shape   [B].
    """

    @abstractmethod
    def forward(
        self,
        market: torch.Tensor,
        account: torch.Tensor,
        globals_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value)."""

    @abstractmethod
    def act(
        self,
        market: torch.Tensor,
        account: torch.Tensor,
        globals_vec: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax-select) an action. Returns (action, log_prob, value)."""

    @abstractmethod
    def evaluate(
        self,
        market: torch.Tensor,
        account: torch.Tensor,
        globals_vec: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a given action. Returns (log_prob, entropy, value)."""
