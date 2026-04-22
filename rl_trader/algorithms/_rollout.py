"""Shared rollout infrastructure for RL algorithms.

Currently: a simple synchronous vectorised env wrapper with auto-reset and
episode-level reward / pnl / length tracking. Future algorithms can reuse this.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..envs.multi_asset import EnvConfig, MultiAssetTradingEnv


class SyncVecEnv:
    """Sequentially-stepped N-environment wrapper with auto-reset.

    Tracks a rolling window of the last 100 completed episodes' rewards, lengths
    and cumulative PnLs for logging purposes.
    """

    def __init__(self, n_envs: int, env_cfg: EnvConfig, base_seed: int = 0):
        self.n = n_envs
        self.envs = [MultiAssetTradingEnv(env_cfg) for _ in range(n_envs)]
        self._base_seed = base_seed

        # per-env live episode accumulators
        self._ep_r   = np.zeros(n_envs, dtype=np.float64)
        self._ep_len = np.zeros(n_envs, dtype=np.int64)
        self._ep_pnl = np.zeros(n_envs, dtype=np.float64)

        # rolling windows for metrics
        self.recent_ep_rewards = deque(maxlen=100)
        self.recent_ep_lengths = deque(maxlen=100)
        self.recent_ep_pnls    = deque(maxlen=100)
        self.recent_ep_term    = deque(maxlen=100)  # term vs trunc

    def reset(self) -> dict[str, np.ndarray]:
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=self._base_seed + i)
            obs.append(o)
        self._ep_r[:] = 0
        self._ep_len[:] = 0
        self._ep_pnl[:] = 0
        return self._stack(obs)

    def step(self, actions: np.ndarray):
        obs_list: list[dict[str, np.ndarray]] = []
        rewards = np.zeros(self.n, dtype=np.float32)
        dones   = np.zeros(self.n, dtype=np.float32)
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            rewards[i] = r
            self._ep_r[i]   += r
            self._ep_len[i] += 1
            self._ep_pnl[i]  = info["cum_pnl"]
            if term or trunc:
                self.recent_ep_rewards.append(float(self._ep_r[i]))
                self.recent_ep_lengths.append(int(self._ep_len[i]))
                self.recent_ep_pnls.append(float(self._ep_pnl[i]))
                self.recent_ep_term.append(bool(term))
                self._ep_r[i] = 0
                self._ep_len[i] = 0
                self._ep_pnl[i] = 0
                o, _ = env.reset()
                dones[i] = 1.0
            obs_list.append(o)
        return self._stack(obs_list), rewards, dones

    def close(self):
        for env in self.envs:
            env.close()

    @staticmethod
    def _stack(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
