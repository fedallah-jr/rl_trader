"""Periodic validation / test evaluation.

`evaluate_policy` runs a set of deterministic rollouts from fixed, evenly-spaced
start indices inside the provided env's date range, and returns aggregate
metrics. Used by the PPO trainer to track the best-so-far model on the val split.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..architectures.base import AbstractActorCritic
from ..envs.multi_asset import EnvConfig, MultiAssetTradingEnv
from .rollout import run_rollout


def evaluate_policy(
    net: AbstractActorCritic,
    env_cfg: EnvConfig,
    *,
    n_episodes: int = 4,
    episode_length: int = 10_080,       # 1 week of 1m bars
    device: str = "cpu",
) -> dict[str, float]:
    """Deterministic rollouts from `n_episodes` evenly-spaced starts inside
    `env_cfg`'s date range.

    Bankruptcy termination is disabled during eval so every rollout runs to a
    fixed length — makes the per-episode PnL directly comparable across evals
    and across models.
    """
    eval_cfg = replace(
        env_cfg,
        episode_min=1,
        episode_max=episode_length,
        bankruptcy_K=None,              # fixed-length rollouts, no early exit
    )
    env = MultiAssetTradingEnv(eval_cfg)

    lo = env._t_lo
    hi = max(lo, env._t_hi - episode_length)
    if n_episodes == 1:
        starts = np.array([lo], dtype=int)
    else:
        starts = np.linspace(lo, hi, n_episodes, dtype=int)

    per_episode_net: list[float] = []
    per_episode_steps: list[int] = []
    all_mean_rewards: list[float] = []

    was_training = net.training
    net.eval()
    try:
        for s in starts:
            res = run_rollout(
                env, net,
                start_idx=int(s),
                n_steps=episode_length,
                deterministic=True,
                device=device,
            )
            net_pnl = float((res.step_pnl - res.fees).sum())
            per_episode_net.append(net_pnl)
            per_episode_steps.append(int(res.rewards.size))
            all_mean_rewards.append(float(res.rewards.mean()))
    finally:
        net.train(was_training)
        env.close()

    return {
        "mean_net_pnl":     float(np.mean(per_episode_net)),
        "std_net_pnl":      float(np.std(per_episode_net)),
        "min_net_pnl":      float(np.min(per_episode_net)),
        "max_net_pnl":      float(np.max(per_episode_net)),
        "mean_step_reward": float(np.mean(all_mean_rewards)),
        "n_episodes":       n_episodes,
        "episode_length":   episode_length,
        "total_steps":      int(sum(per_episode_steps)),
    }
