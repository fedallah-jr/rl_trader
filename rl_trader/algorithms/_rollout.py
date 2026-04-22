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

    `step` returns `(obs, rewards, terminations, truncations, final_obs)`:
      * `terminations[i] = 1.0` iff env i reached a true terminal state
        (e.g. bankruptcy). V(s_terminal) = 0 by definition.
      * `truncations[i]  = 1.0` iff env i hit an external cutoff (time limit,
        end of data range) — the underlying MDP did not end, we just stopped
        observing. The learner must bootstrap V(s') from the *pre-reset* obs.
      * If both fire on the same step, term dominates (truncations[i] = 0).
      * `final_obs[i]` holds the pre-reset observation iff env i ended its
        episode (term or trunc), and `None` otherwise. Needed so that PPO
        can compute V(final_obs) for truncation bootstrap before the auto-
        reset overwrites the observation stream.
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
        rewards       = np.zeros(self.n, dtype=np.float32)
        terminations  = np.zeros(self.n, dtype=np.float32)
        truncations   = np.zeros(self.n, dtype=np.float32)
        final_obs: list[dict[str, np.ndarray] | None] = [None] * self.n

        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            rewards[i] = r
            # Gymnasium convention: if both fire, term dominates — V=0
            # on a true terminal; no bootstrap to recover.
            terminations[i] = float(term)
            truncations[i]  = float(trunc and not term)

            self._ep_r[i]   += r
            self._ep_len[i] += 1
            self._ep_pnl[i]  = info["cum_pnl"]
            if term or trunc:
                # Capture the pre-reset observation so the caller can compute
                # V(final_obs) as a truncation bootstrap. For pure
                # terminations this obs is never consumed (mask kills it in
                # GAE) but we capture uniformly for simplicity.
                final_obs[i] = o
                self.recent_ep_rewards.append(float(self._ep_r[i]))
                self.recent_ep_lengths.append(int(self._ep_len[i]))
                self.recent_ep_pnls.append(float(self._ep_pnl[i]))
                self.recent_ep_term.append(bool(term))
                self._ep_r[i] = 0
                self._ep_len[i] = 0
                self._ep_pnl[i] = 0
                o, _ = env.reset()
            obs_list.append(o)

        return self._stack(obs_list), rewards, terminations, truncations, final_obs

    def close(self):
        for env in self.envs:
            env.close()

    @staticmethod
    def _stack(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
