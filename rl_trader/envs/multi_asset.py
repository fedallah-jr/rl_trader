"""Gymnasium environment for multi-asset 1m trading on 6 symbols.

Action: MultiDiscrete([3]*6), per symbol in {0, 1, 2} → target sign {-1, 0, +1}.

Semantics:
  - Decision made at close of bar t, fill at open of bar t+1.
  - Fixed notional U per position (default 1.0 after reward-normalization).
  - Same action as prior step => no trade, no fee (idempotent).
  - Sign flip => close old + open new at open[t+1], paying 2 × taker_bps × U.

Reward: `(step_pnl - fees) / U`, i.e. return on unit notional.

Observation (Dict):
  market  : float32 [window, 6, F]  past `window` feature vectors up to & including t
  account : float32 [6, 3]          per-symbol {sign, unrealized_ret_i, log1p(time_in_pos)}
  globals : float32 [5]             sin/cos hour, sin/cos day-of-week, tanh(cum_pnl/U)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ._common import load_features_cached


@dataclass
class EnvConfig:
    features_path: str | Path = "dataset/features/features.npz"
    window: int = 128                   # bars of history in observation
    episode_min: int = 1440             # min steps per episode (1 day)
    episode_max: int = 10080            # max steps per episode (1 week)
    taker_bps: float = 5e-4             # 5 bps all-in per leg (taker + slippage)
    notional_U: float = 1.0             # fixed per-symbol notional, reward normalized by this
    start_ts: str | None = None         # inclusive ISO date, e.g. "2022-01-01"
    end_ts:   str | None = None         # inclusive ISO date
    bankruptcy_K: float | None = 3.0    # terminate when cum_pnl < -K × U; None to disable
    bankruptcy_penalty: float = -1.0    # reward delivered on bankruptcy termination

    # Vol-scaled reward (Zhang, Zohren, Roberts 2019, option A adaptation).
    # Per-symbol reward contribution is multiplied by `sigma_target / σ_i(t)` so
    # symbols with different volatilities contribute equally to gradient signal.
    # Positions, fees, and cum_pnl (raw) are all UNCHANGED — only the learning
    # reward is reweighted. Set sigma_target=None to disable.
    sigma_target: float | None = 1e-3   # ≈ 10 bps per 1m bar (mean across our 6 syms)
    sigma_min: float = 1e-5             # lower clamp on σ_i to bound the weight


class MultiAssetTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()

        d = load_features_cached(self.cfg.features_path)
        self._features = d["features"]           # [T_all, 6, F]
        self._open     = d["open"].astype(np.float64, copy=False)
        self._close    = d["close"].astype(np.float64, copy=False)
        self._ts_ns    = d["timestamps"].astype(np.int64, copy=False)
        self._feature_names = [str(x) for x in d["feature_names"]]
        self._symbols  = [str(x) for x in d["symbols"]]

        self.T_all, self.N, self.F = self._features.shape
        assert self.N == 6
        assert self._open.shape == (self.T_all, self.N)
        assert self._close.shape == (self.T_all, self.N)

        # locate rvol_60m for vol-scaled reward; feature_names must include it.
        if self.cfg.sigma_target is not None:
            try:
                self._rvol_idx = self._feature_names.index("rvol_60m")
            except ValueError as e:
                raise ValueError(
                    "EnvConfig.sigma_target is set but feature tensor has no "
                    "'rvol_60m' column; rebuild features or pass sigma_target=None."
                ) from e
        else:
            self._rvol_idx = -1

        self._t_lo, self._t_hi = self._resolve_range(self.cfg.start_ts, self.cfg.end_ts)

        # cyclic time features, derived once for the full dataset
        secs = self._ts_ns // 1_000_000_000
        minutes = (secs // 60) % 1440
        dow = ((secs // 86400) + 4) % 7                     # 1970-01-01 is Thursday (4)
        two_pi_min = 2 * np.pi * minutes / 1440.0
        two_pi_dow = 2 * np.pi * dow / 7.0
        self._sin_min = np.sin(two_pi_min).astype(np.float32)
        self._cos_min = np.cos(two_pi_min).astype(np.float32)
        self._sin_dow = np.sin(two_pi_dow).astype(np.float32)
        self._cos_dow = np.cos(two_pi_dow).astype(np.float32)

        self.observation_space = spaces.Dict({
            "market":  spaces.Box(-np.inf, np.inf, (self.cfg.window, self.N, self.F), np.float32),
            "account": spaces.Box(-np.inf, np.inf, (self.N, 3), np.float32),
            "globals": spaces.Box(-np.inf, np.inf, (5,), np.float32),
        })
        self.action_space = spaces.MultiDiscrete([3] * self.N)

        # per-episode state (initialized in reset)
        self._rng: np.random.Generator | None = None
        self._step_idx: int = 0
        self._episode_start: int = 0
        self._episode_length: int = 0
        self._steps_taken: int = 0
        self._position_sign = np.zeros(self.N, dtype=np.int8)
        self._entry_price  = np.zeros(self.N, dtype=np.float64)
        self._entry_step   = np.full(self.N, -1, dtype=np.int64)
        self._realized_pnl: float = 0.0
        self._unrealized_at_close = np.zeros(self.N, dtype=np.float64)  # per-symbol
        self._cum_pnl: float = 0.0

    # ---- range helpers ------------------------------------------------

    def _resolve_range(self, start_ts: str | None, end_ts: str | None) -> tuple[int, int]:
        lo = 0
        hi = self.T_all - 1
        if start_ts is not None:
            t0_ns = np.datetime64(start_ts, "ns").view("i8")
            lo = int(np.searchsorted(self._ts_ns, t0_ns, side="left"))
        if end_ts is not None:
            t1_ns = np.datetime64(end_ts, "ns").view("i8")
            hi = int(np.searchsorted(self._ts_ns, t1_ns, side="right")) - 1
        lo = max(lo, self.cfg.window - 1)
        hi = min(hi, self.T_all - 2)
        if hi - lo < self.cfg.episode_min:
            raise ValueError(f"requested range [{start_ts}, {end_ts}] too small "
                             f"(usable={hi-lo+1}, min episode={self.cfg.episode_min})")
        return lo, hi

    # ---- gym API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        options = options or {}
        if "start_idx" in options:
            start = int(options["start_idx"])
        else:
            start = int(self._rng.integers(self._t_lo, self._t_hi - self.cfg.episode_min + 1))

        if "episode_length" in options:
            ep_len = int(options["episode_length"])
        else:
            ep_len = int(self._rng.integers(self.cfg.episode_min, self.cfg.episode_max + 1))

        ep_len = min(ep_len, self._t_hi - start)
        if ep_len < 1:
            raise RuntimeError("derived episode_length < 1; check range/config")

        self._step_idx = start
        self._episode_start = start
        self._episode_length = ep_len
        self._steps_taken = 0
        self._position_sign[:] = 0
        self._entry_price[:]  = 0.0
        self._entry_step[:]   = -1
        self._realized_pnl = 0.0
        self._unrealized_at_close[:] = 0.0
        self._cum_pnl = 0.0

        obs = self._get_observation()
        info = {
            "start_idx": start,
            "episode_length": ep_len,
            "start_ts_ns": int(self._ts_ns[start]),
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self._rng is None:
            raise RuntimeError("must call reset() before step()")

        action = np.asarray(action, dtype=np.int64).reshape(self.N)
        if ((action < 0) | (action > 2)).any():
            raise ValueError(f"action out of range: {action}")
        new_sign = (action - 1).astype(np.int8)

        t = self._step_idx
        t_next = t + 1
        open_next  = self._open[t_next]
        close_next = self._close[t_next]

        unrealized_old = self._unrealized_at_close.copy()       # per-symbol snapshot
        U = self.cfg.notional_U
        fee_per_leg = U * self.cfg.taker_bps

        fees = np.zeros(self.N, dtype=np.float64)               # per-symbol
        realized_delta = np.zeros(self.N, dtype=np.float64)     # per-symbol

        for i in range(self.N):
            old = int(self._position_sign[i])
            ns  = int(new_sign[i])
            if ns == old:
                continue
            if old != 0:
                realized_delta[i] += old * U * (open_next[i] / self._entry_price[i] - 1.0)
                fees[i] += fee_per_leg
            if ns != 0:
                self._position_sign[i] = ns
                self._entry_price[i]   = float(open_next[i])
                self._entry_step[i]    = t_next
                fees[i] += fee_per_leg
            else:
                self._position_sign[i] = 0
                self._entry_price[i]   = 0.0
                self._entry_step[i]    = -1

        self._realized_pnl += float(realized_delta.sum())

        unrealized_new = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            if self._position_sign[i] != 0:
                unrealized_new[i] = (
                    int(self._position_sign[i]) * U
                    * (close_next[i] / self._entry_price[i] - 1.0)
                )

        step_pnl = realized_delta + (unrealized_new - unrealized_old)   # per-symbol
        step_net = step_pnl - fees                                      # per-symbol

        # Per-symbol reward weights: σ_tgt / σ_i(t) with σ_i clamped, or 1.0 if
        # vol scaling is disabled. σ_i is ex-ante (taken from features at time
        # `t` — the decision point — so strictly causal for the t→t+1 bar).
        if self.cfg.sigma_target is not None:
            sigma_i = np.asarray(
                self._features[t, :, self._rvol_idx], dtype=np.float64
            )
            sigma_safe = np.maximum(sigma_i, self.cfg.sigma_min)
            weights = self.cfg.sigma_target / sigma_safe                 # [N]
        else:
            weights = np.ones(self.N, dtype=np.float64)

        reward = float((step_net * weights).sum() / U)

        self._unrealized_at_close = unrealized_new
        self._cum_pnl += float(step_net.sum())                  # raw PnL (unscaled)
        self._step_idx = t_next
        self._steps_taken += 1

        terminated = False
        truncated = False

        if self.cfg.bankruptcy_K is not None and self._cum_pnl < -self.cfg.bankruptcy_K * U:
            reward += self.cfg.bankruptcy_penalty
            terminated = True

        if self._steps_taken >= self._episode_length:
            truncated = True
        if self._step_idx >= self._t_hi:
            truncated = True

        obs = self._get_observation()
        info = {
            "step_pnl": float(step_pnl.sum()),                  # RAW aggregate
            "step_pnl_per_symbol": step_pnl,                    # RAW per-symbol
            "fees": float(fees.sum()),                          # RAW aggregate
            "fees_per_symbol": fees,                            # RAW per-symbol
            "realized_pnl": self._realized_pnl,
            "unrealized": float(unrealized_new.sum()),
            "cum_pnl": self._cum_pnl,                           # RAW cumulative
            "position_sign": self._position_sign.copy(),
            "step_idx": self._step_idx,
            "ts_ns": int(self._ts_ns[self._step_idx]),
            "reward_weights": weights,                          # σ_tgt / σ_i
        }
        return obs, float(reward), terminated, truncated, info

    # ---- observation -------------------------------------------------

    def _get_observation(self) -> dict[str, np.ndarray]:
        t = self._step_idx
        w = self.cfg.window
        market = self._features[t - w + 1 : t + 1]

        sign_f = self._position_sign.astype(np.float32)
        in_pos = self._position_sign != 0
        ur_ret = np.zeros(self.N, dtype=np.float32)
        if in_pos.any():
            ur_ret[in_pos] = (
                self._position_sign[in_pos]
                * (self._close[t, in_pos] / self._entry_price[in_pos] - 1.0)
            ).astype(np.float32)
        time_in_pos = np.where(
            in_pos, np.log1p(t - self._entry_step).astype(np.float32), 0.0
        ).astype(np.float32)
        account = np.stack([sign_f, ur_ret, time_in_pos], axis=1)

        globals_vec = np.array(
            [
                self._sin_min[t], self._cos_min[t],
                self._sin_dow[t], self._cos_dow[t],
                np.tanh(self._cum_pnl / self.cfg.notional_U).astype(np.float32),
            ],
            dtype=np.float32,
        )

        return {
            "market":  np.ascontiguousarray(market),
            "account": account,
            "globals": globals_vec,
        }

    # ---- misc ---------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def close(self):
        pass
