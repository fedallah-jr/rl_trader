"""Periodic validation / test evaluation.

`evaluate_policy` runs a set of deterministic rollouts from fixed, evenly-spaced
start indices inside the provided env's date range, and returns aggregate
metrics. Used by the PPO trainer to track the best-so-far model on the val split.

All `n_episodes` rollouts are stepped in lockstep as a single batched rollout
(B = n_episodes) so the policy forward runs once per step instead of once per
(step × episode). Safe because bankruptcy termination is disabled for eval
and `episode_length` is fixed — every env truncates at exactly the same step.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .. import resolve_device
from ..architectures.base import AbstractActorCritic
from ..envs.multi_asset import EnvConfig, MultiAssetTradingEnv


def evaluate_policy(
    net: AbstractActorCritic,
    env_cfg: EnvConfig,
    *,
    n_episodes: int = 4,
    episode_length: int = 10_080,       # 1 week of 1m bars
    device: str | torch.device = "auto",
) -> dict[str, float]:
    """Deterministic batched rollouts from `n_episodes` evenly-spaced starts
    inside `env_cfg`'s date range.

    Bankruptcy termination is disabled during eval so every rollout runs to a
    fixed length — makes the per-episode PnL directly comparable across evals
    and across models, and lets all envs step in lockstep under one batched
    forward per step.
    """
    dev = resolve_device(device)
    eval_cfg = replace(
        env_cfg,
        episode_min=1,
        episode_max=episode_length,
        bankruptcy_K=None,              # fixed-length rollouts, no early exit
    )
    envs = [MultiAssetTradingEnv(eval_cfg) for _ in range(n_episodes)]

    lo = envs[0]._t_lo
    hi = max(lo, envs[0]._t_hi - episode_length)
    if n_episodes == 1:
        starts = np.array([lo], dtype=int)
    else:
        starts = np.linspace(lo, hi, n_episodes, dtype=int)

    was_training = net.training
    net.eval()
    try:
        obs_per_env: list[dict[str, np.ndarray]] = []
        for env, s in zip(envs, starts):
            obs, _ = env.reset(seed=0, options={
                "start_idx": int(s),
                "episode_length": int(episode_length),
            })
            obs_per_env.append(obs)

        net_pnl    = np.zeros(n_episodes, dtype=np.float64)
        reward_sum = np.zeros(n_episodes, dtype=np.float64)

        for _ in range(episode_length):
            market  = np.stack([o["market"]  for o in obs_per_env], axis=0)
            account = np.stack([o["account"] for o in obs_per_env], axis=0)
            gl      = np.stack([o["globals"] for o in obs_per_env], axis=0)

            m_t = torch.as_tensor(market,  device=dev)
            a_t = torch.as_tensor(account, device=dev)
            g_t = torch.as_tensor(gl,      device=dev)
            with torch.no_grad():
                action, _, _ = net.act(m_t, a_t, g_t, deterministic=True)
            action_np = action.cpu().numpy()           # [B, N]

            next_obs: list[dict[str, np.ndarray]] = []
            for i, env in enumerate(envs):
                o, r, _term, _trunc, info = env.step(action_np[i])
                net_pnl[i]    += info["step_pnl"] - info["fees"]
                reward_sum[i] += r
                next_obs.append(o)
            obs_per_env = next_obs
    finally:
        net.train(was_training)
        for env in envs:
            env.close()

    return {
        "mean_net_pnl":     float(np.mean(net_pnl)),
        "std_net_pnl":      float(np.std(net_pnl)),
        "min_net_pnl":      float(np.min(net_pnl)),
        "max_net_pnl":      float(np.max(net_pnl)),
        "mean_step_reward": float(np.mean(reward_sum / episode_length)),
        "n_episodes":       n_episodes,
        "episode_length":   episode_length,
        "total_steps":      int(n_episodes * episode_length),
    }
