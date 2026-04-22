"""Deterministic rollout of a trained policy in an env, logging every step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..architectures.base import AbstractActorCritic
from ..architectures.factored_attention import ActorCritic, PolicyConfig
from ..envs.multi_asset import MultiAssetTradingEnv


@dataclass
class RolloutResult:
    timestamps: np.ndarray        # [T] ns since epoch
    actions:    np.ndarray        # [T, 6] in {-1, 0, +1}
    positions:  np.ndarray        # [T, 6] in {-1, 0, +1}
    rewards:    np.ndarray        # [T]
    step_pnl:   np.ndarray        # [T]
    fees:       np.ndarray        # [T]
    cum_pnl:    np.ndarray        # [T] cumulative net pnl (PnL − fees)
    symbols:    list[str]


def run_rollout(
    env: MultiAssetTradingEnv,
    net: AbstractActorCritic,
    *,
    start_idx: int | None = None,
    n_steps: int | None = None,
    deterministic: bool = True,
    device: str | torch.device = "cpu",
) -> RolloutResult:
    """Reset the env once, roll out the policy for `n_steps` steps (or a full
    episode), and record every action/position/reward/pnl."""
    device = torch.device(device)
    net = net.eval().to(device)

    options: dict[str, Any] = {}
    if start_idx is not None:
        options["start_idx"] = int(start_idx)
    if n_steps is not None:
        options["episode_length"] = int(n_steps)
    obs, info = env.reset(seed=0, options=options)
    ep_len = info["episode_length"]

    ts_all, acts_all, pos_all, rews_all, pnl_all, fee_all = [], [], [], [], [], []

    for _ in range(ep_len):
        m = torch.as_tensor(obs["market"],  device=device).unsqueeze(0)
        a = torch.as_tensor(obs["account"], device=device).unsqueeze(0)
        g = torch.as_tensor(obs["globals"], device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = net.act(m, a, g, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        ts_all.append(info["ts_ns"])
        acts_all.append((action_np - 1).copy())
        pos_all.append(info["position_sign"].copy())
        rews_all.append(reward)
        pnl_all.append(info["step_pnl"])
        fee_all.append(info["fees"])

        if terminated or truncated:
            break

    step_pnl = np.asarray(pnl_all, dtype=np.float64)
    fees     = np.asarray(fee_all, dtype=np.float64)
    return RolloutResult(
        timestamps=np.asarray(ts_all, dtype=np.int64),
        actions=np.asarray(acts_all, dtype=np.int8),
        positions=np.asarray(pos_all, dtype=np.int8),
        rewards=np.asarray(rews_all, dtype=np.float64),
        step_pnl=step_pnl,
        fees=fees,
        cum_pnl=np.cumsum(step_pnl - fees),
        symbols=env.symbols,
    )


def load_policy(ckpt_path: str | Path, device: str | torch.device = "cpu") -> ActorCritic:
    """Load a PPO checkpoint and reconstruct the ActorCritic from stored cfg."""
    dev = torch.device(device)
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    pcfg_dict = ck.get("policy_cfg") or {}
    pcfg = PolicyConfig(**pcfg_dict) if pcfg_dict else PolicyConfig()
    net = ActorCritic(pcfg).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net
