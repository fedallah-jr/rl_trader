"""Factored space-time attention actor-critic for the multi-asset trader.

Architecture mirrors the design doc:

    market[B, T, 6, F]
        -> Linear(F->D) + time_pos_emb + symbol_emb
        -> L × (time-attn causal within symbol  then  cross-asset attn across symbols)
        -> attention-pool over T
        -> z[B, 6, D]

Actor (shared per symbol):
    head([z_i ; account_i ; globals]) -> 3 logits   ->   MultiDiscrete([3]*6)

Critic:
    mean-pool z over symbols, concat globals, MLP -> V(s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import AbstractActorCritic


# ---- config --------------------------------------------------------------

@dataclass
class PolicyConfig:
    n_symbols: int = 6
    n_features: int = 17
    n_account: int = 3
    n_globals: int = 5
    n_actions: int = 3
    window: int = 128
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    head_hidden: int = 64


# ---- building blocks -----------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: MHA then FFN, both with residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln_attn(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln_ffn(x))
        return x


class FactoredBlock(nn.Module):
    """Time attention within each symbol, then cross-asset attention at each time."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.time_block = TransformerBlock(d_model, n_heads, ffn_mult, dropout)
        self.sym_block  = TransformerBlock(d_model, n_heads, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D]
        B, N, T, D = x.shape

        # time attn: treat each symbol as its own sequence of length T
        x_t = x.reshape(B * N, T, D)
        x_t = self.time_block(x_t, attn_mask=causal_mask)
        x = x_t.reshape(B, N, T, D)

        # symbol attn: at each timestep, 6 symbols attend to each other
        x_s = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x_s = self.sym_block(x_s, attn_mask=None)
        x = x_s.reshape(B, T, N, D).permute(0, 2, 1, 3).contiguous()
        return x


class TimeAttentionPool(nn.Module):
    """Pool the time dimension with a learnable query attending to the sequence."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D] -> [B, N, D]
        B, N, T, D = x.shape
        x_f = x.reshape(B * N, T, D)
        k = self.ln(x_f)
        q = self.query.expand(B * N, -1, -1)
        h, _ = self.attn(q, k, k, need_weights=False)
        return h.reshape(B, N, D)


# ---- full actor-critic ---------------------------------------------------

class ActorCritic(AbstractActorCritic):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or PolicyConfig()

        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.time_pos_emb = nn.Parameter(torch.zeros(1, 1, cfg.window, cfg.d_model))
        self.symbol_emb   = nn.Parameter(torch.zeros(1, cfg.n_symbols, 1, cfg.d_model))
        nn.init.normal_(self.time_pos_emb, std=0.02)
        nn.init.normal_(self.symbol_emb,   std=0.02)

        causal = torch.triu(torch.ones(cfg.window, cfg.window), diagonal=1).bool()
        self.register_buffer("causal_mask", causal, persistent=False)

        self.blocks = nn.ModuleList(
            [FactoredBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.pool = TimeAttentionPool(cfg.d_model, n_heads=cfg.n_heads)

        head_in_actor  = cfg.d_model + cfg.n_account + cfg.n_globals
        head_in_critic = cfg.d_model + cfg.n_globals

        self.actor_head = nn.Sequential(
            nn.Linear(head_in_actor, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, cfg.n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(head_in_critic, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Actor last layer: tiny gain so initial logits ≈ 0 (near-uniform).
        final = self.actor_head[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)
        # Small positive bias on the "flat" action (index 1 in {-1,0,+1}) so
        # the agent starts biased toward do-nothing and doesn't bleed fees.
        with torch.no_grad():
            final.bias[1] = 0.5

        # Critic last layer: unit gain, zero bias.
        fc = self.critic_head[-1]
        nn.init.orthogonal_(fc.weight, gain=1.0)
        nn.init.zeros_(fc.bias)

    # ---- encoder --------------------------------------------------------

    def encode(self, market: torch.Tensor) -> torch.Tensor:
        """market: [B, T, N, F] (as emitted by the env) -> z: [B, N, D]."""
        if market.dim() != 4:
            raise ValueError(f"market must be 4D [B, T, N, F], got {tuple(market.shape)}")
        B, T, N, F = market.shape
        assert N == self.cfg.n_symbols and F == self.cfg.n_features
        # permute to [B, N, T, F]
        x = market.permute(0, 2, 1, 3).contiguous()
        x = self.input_proj(x)                              # [B, N, T, D]
        x = x + self.time_pos_emb[:, :, :T, :]              # broadcast time
        x = x + self.symbol_emb                             # broadcast symbol
        for block in self.blocks:
            x = block(x, causal_mask=self.causal_mask[:T, :T])
        z = self.pool(x)                                    # [B, N, D]
        return z

    # ---- forward / sample / evaluate -----------------------------------

    def forward(
        self,
        market: torch.Tensor,
        account: torch.Tensor,
        globals_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(market)                             # [B, N, D]
        B, N, D = z.shape
        g_exp = globals_vec.unsqueeze(1).expand(-1, N, -1)  # [B, N, G]
        h = torch.cat([z, account, g_exp], dim=-1)          # [B, N, D+A+G]
        logits = self.actor_head(h)                         # [B, N, n_actions]

        z_pool = z.mean(dim=1)                              # [B, D]
        v_in = torch.cat([z_pool, globals_vec], dim=-1)     # [B, D+G]
        value = self.critic_head(v_in).squeeze(-1)          # [B]
        return logits, value

    @torch.no_grad()
    def act(
        self,
        market: torch.Tensor,
        account: torch.Tensor,
        globals_vec: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(market, account, globals_vec)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)        # [B] (joint over N dims)
        return action, log_prob, value

    def evaluate(
        self,
        market: torch.Tensor,
        account: torch.Tensor,
        globals_vec: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(market, account, globals_vec)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action).sum(dim=-1)        # [B]
        entropy  = dist.entropy().sum(dim=-1)               # [B]
        return log_prob, entropy, value

    # ---- util -----------------------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
