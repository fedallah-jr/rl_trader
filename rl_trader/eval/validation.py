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

    If the date range is too short to fit the requested `episode_length`, the
    env caps each episode to the usable length on reset; we honour that by
    looping `min(per-env actual lengths)` times and reporting that length in
    the returned metrics, so best-model selection is never based on a fake
    trajectory extent.
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
        per_env_ep_len: list[int] = []
        for env, s in zip(envs, starts):
            obs, info = env.reset(seed=0, options={
                "start_idx": int(s),
                "episode_length": int(episode_length),
            })
            obs_per_env.append(obs)
            per_env_ep_len.append(int(info["episode_length"]))

        # Env.reset caps ep_len to `_t_hi - start` when the date range can't
        # fit the requested length. Lockstep batching forces us to use the
        # min — any shorter env would otherwise be stepped past its episode
        # boundary (wrong metrics + eventual out-of-range obs).
        actual_ep_len = int(min(per_env_ep_len))
        if actual_ep_len < episode_length:
            print(
                f"  WARN: eval range truncates episode_length "
                f"{episode_length} -> {actual_ep_len} "
                f"(per-env lens: {per_env_ep_len})"
            )

        net_pnl     = np.zeros(n_episodes, dtype=np.float64)
        reward_sum  = np.zeros(n_episodes, dtype=np.float64)
        steps_taken = 0

        for _ in range(actual_ep_len):
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
            any_done = False
            for i, env in enumerate(envs):
                o, r, term, trunc, info = env.step(action_np[i])
                net_pnl[i]    += info["step_pnl"] - info["fees"]
                reward_sum[i] += r
                next_obs.append(o)
                if term or trunc:
                    any_done = True
            obs_per_env = next_obs
            steps_taken += 1
            # Honour lockstep termination. With bankruptcy_K=None and the
            # start schedule above this only fires on the scheduled final
            # step (trunc=True) — still worth breaking on so that future
            # env changes (e.g. bankruptcy reintroduced) can't silently
            # corrupt per-episode metrics.
            if any_done:
                break
    finally:
        net.train(was_training)
        for env in envs:
            env.close()

    step_divisor = max(steps_taken, 1)
    return {
        "mean_net_pnl":     float(np.mean(net_pnl)),
        "std_net_pnl":      float(np.std(net_pnl)),
        "min_net_pnl":      float(np.min(net_pnl)),
        "max_net_pnl":      float(np.max(net_pnl)),
        "mean_step_reward": float(np.mean(reward_sum / step_divisor)),
        "n_episodes":       n_episodes,
        "episode_length":   steps_taken,
        "total_steps":      int(n_episodes * steps_taken),
    }
