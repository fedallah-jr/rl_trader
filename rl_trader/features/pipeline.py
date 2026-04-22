"""Feature precompute pipeline.

Turns the 6 per-symbol kline pickles and 6 per-symbol metric pickles into a
single aligned feature tensor usable by the environment.

Contract:
  Output .npz has, after dropping warm-up and gap-affected rows:
    features      : float32  [T, 6, F=17]
    open          : float64  [T, 6]             fill prices
    close         : float64  [T, 6]             for PnL marking
    timestamps    : int64    [T]                ns-since-epoch (UTC)
    feature_names : str      [F]
    symbols       : str      [6]
  And is guaranteed NaN-free across `features`, `open`, `close`.

Causality notes:
- Multi-horizon log returns, intrabar shape, rolling stats all use only data
  at or before time t.
- Metric rows are merged with a 3-minute look-back shift (empirically verified
  publication lag is ~1-2 min; 3 min gives headroom). At kline minute t the
  applicable metric stamp is the most recent `S <= t - 3 min`.
- Rows inside a 1440-bar window of any 1m gap (SOL/XRP missing days) are
  dropped so no feature straddles discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import SYMBOLS


# ---- configuration -------------------------------------------------------

METRIC_SHIFT_MINUTES = 3          # publication-delay safety margin
ROLLING_WINDOW        = 1440      # 24h of 1m bars for z-scores
WARMUP_BARS           = 1440      # longest lookback horizon
Z_CLIP                = 5.0       # clip rolling z-scores to +/- this

METRIC_COLS_USED = (
    "sum_open_interest",
    "sum_open_interest_value",   # kept for possible future use; not in feature list
    "count_long_short_ratio",
)

FEATURE_NAMES = (
    "r_1m", "r_5m", "r_15m", "r_1h", "r_4h", "r_1d",
    "hl_range", "co_range",
    "log_volume_z", "log_count_z", "taker_imbalance",
    "rvol_60m",
    "d_log_oi_5m", "d_log_oi_1h", "d_log_oi_4h",
    "log_ls_ratio",
    "metric_staleness_mins",
)
F = len(FEATURE_NAMES)  # 17


# ---- loading -------------------------------------------------------------

def load_klines(processed_root: Path, symbol: str) -> pd.DataFrame:
    kl = pd.read_pickle(processed_root / "klines" / f"{symbol}_1m.pkl")
    # Dataset is already sorted/deduped; reconfirm and coerce dtypes we need.
    kl = kl.sort_index()
    # Belt-and-suspenders: any NaN would propagate silently through features.
    assert not kl[["open", "high", "low", "close", "volume",
                   "count", "taker_buy_volume"]].isna().any().any(), \
        f"kline NaN in {symbol}"
    return kl


def load_metrics(processed_root: Path, symbol: str) -> pd.DataFrame:
    m = pd.read_pickle(processed_root / "metrics" / f"{symbol}_metrics.pkl")
    m = m.sort_index()
    # Only keep the columns we actually use.
    m = m[list(METRIC_COLS_USED)].copy()
    # Forward-fill the thin gaps we characterized:
    #   count_long_short_ratio: ~1% scattered NaN
    #   sum_open_interest / _value: ~0.05% zero anomalies (treat zero as missing then ffill)
    m["sum_open_interest"] = m["sum_open_interest"].where(m["sum_open_interest"] > 0).ffill()
    m["sum_open_interest_value"] = (
        m["sum_open_interest_value"].where(m["sum_open_interest_value"] > 0).ffill()
    )
    m["count_long_short_ratio"] = m["count_long_short_ratio"].ffill()
    return m


# ---- metric alignment to 1m index ----------------------------------------

def align_metrics_to_klines(
    klines_index: pd.DatetimeIndex,
    metrics: pd.DataFrame,
    shift_minutes: int = METRIC_SHIFT_MINUTES,
) -> pd.DataFrame:
    """Return a DataFrame indexed by `klines_index` holding, for each kline
    time t, the most recent metric row with `stamp <= t - shift_minutes`,
    plus a `metric_staleness_mins` column = t - stamp in minutes.
    """
    shift = pd.Timedelta(minutes=shift_minutes)

    m = metrics.copy()
    m.index.name = "create_time"
    m_reset = m.reset_index().sort_values("create_time")

    k = pd.DataFrame(
        {"t_kline": klines_index, "t_query": klines_index - shift}
    ).sort_values("t_query").reset_index(drop=True)

    merged = pd.merge_asof(
        k,
        m_reset,
        left_on="t_query",
        right_on="create_time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["metric_staleness_mins"] = (
        (merged["t_kline"] - merged["create_time"]).dt.total_seconds() / 60.0
    )
    merged = merged.sort_values("t_kline").set_index("t_kline")
    merged.index.name = klines_index.name
    return merged


# ---- per-symbol feature computation --------------------------------------

def _z(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std()
    z = (series - mu) / sd
    return z.clip(lower=-Z_CLIP, upper=Z_CLIP)


def compute_per_symbol_features(kl: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """kl: klines on the aligned 1m index. m: aligned metrics (same index)."""
    if not kl.index.equals(m.index):
        raise ValueError("kl and m must share the same index")

    out = pd.DataFrame(index=kl.index)
    log_close = np.log(kl["close"].astype(np.float64))

    # multi-horizon log returns
    out["r_1m"]  = log_close.diff(1)
    out["r_5m"]  = log_close.diff(5)
    out["r_15m"] = log_close.diff(15)
    out["r_1h"]  = log_close.diff(60)
    out["r_4h"]  = log_close.diff(240)
    out["r_1d"]  = log_close.diff(1440)

    # intrabar shape
    out["hl_range"] = (kl["high"] - kl["low"]) / kl["close"]
    out["co_range"] = (kl["close"] - kl["open"]) / kl["open"]

    # activity / volume
    log_vol = np.log1p(kl["volume"].clip(lower=0))
    log_cnt = np.log1p(kl["count"].astype(np.float64).clip(lower=0))
    out["log_volume_z"] = _z(log_vol)
    out["log_count_z"]  = _z(log_cnt)

    # taker imbalance, centered
    vol_safe = kl["volume"].where(kl["volume"] > 0)
    taker_imb = (kl["taker_buy_volume"] / vol_safe).fillna(0.5)
    out["taker_imbalance"] = (taker_imb - 0.5).clip(-0.5, 0.5)

    # realized volatility of 1m returns
    out["rvol_60m"] = out["r_1m"].rolling(60, min_periods=60).std()

    # OI deltas (5m, 1h, 4h) on the 1m-aligned OI series — diff at position
    # lag k kline-minutes == k-minute OI change (with 3-min safety shift).
    log_oi = np.log(m["sum_open_interest"].astype(np.float64))
    out["d_log_oi_5m"] = log_oi.diff(5)
    out["d_log_oi_1h"] = log_oi.diff(60)
    out["d_log_oi_4h"] = log_oi.diff(240)

    # long/short ratio, logged for symmetry
    lsr = m["count_long_short_ratio"].astype(np.float64).clip(lower=1e-6)
    out["log_ls_ratio"] = np.log(lsr)

    # staleness from alignment
    out["metric_staleness_mins"] = m["metric_staleness_mins"]

    # ensure we emit columns in declared order
    return out[list(FEATURE_NAMES)]


# ---- gap handling --------------------------------------------------------

def find_gap_safe_mask(index: pd.DatetimeIndex, lookback: int = WARMUP_BARS) -> np.ndarray:
    """Mark rows whose preceding `lookback` bars are all contiguous 1-minute.

    Returns a boolean array over `index`; True = the row's full lookback
    history is gap-free.
    """
    dt = index.to_series().diff().dt.total_seconds()
    # First row's dt is NaN; treat as a gap (no prior history).
    is_gap_row = (dt.fillna(np.inf) > 60.0)
    has_recent_gap = (
        is_gap_row.rolling(lookback, min_periods=1).max().fillna(True).astype(bool)
    )
    return ~has_recent_gap.to_numpy()


# ---- top-level dataset build --------------------------------------------

@dataclass
class BuildSummary:
    n_rows: int
    n_symbols: int
    n_features: int
    start: pd.Timestamp
    end: pd.Timestamp
    dropped_warmup: int
    dropped_gaps: int
    dropped_nan_rows: int


def build_dataset(processed_root: Path, out_path: Path) -> BuildSummary:
    processed_root = Path(processed_root)
    out_path = Path(out_path)

    klines = {s: load_klines(processed_root, s) for s in SYMBOLS}

    # inner-join on the 1m index
    common: pd.DatetimeIndex | None = None
    for s in SYMBOLS:
        idx = klines[s].index
        common = idx if common is None else common.intersection(idx)
    assert common is not None and len(common) > 0
    common = common.sort_values()

    n_all = len(common)

    # align metrics per symbol on the common index, then compute features
    per_symbol_features: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        kl = klines[s].loc[common]
        m_raw = load_metrics(processed_root, s)
        m_aln = align_metrics_to_klines(common, m_raw)
        per_symbol_features[s] = compute_per_symbol_features(kl, m_aln)

    # gap mask on the common index
    gap_safe = find_gap_safe_mask(common, lookback=WARMUP_BARS)
    dropped_gaps = int((~gap_safe).sum())

    # stack to [T, 6, F]
    feats = np.stack(
        [per_symbol_features[s][list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
         for s in SYMBOLS],
        axis=1,
    )  # shape [T, 6, F]

    opens  = np.stack([klines[s].loc[common, "open"].to_numpy(dtype=np.float64)
                       for s in SYMBOLS], axis=1)
    closes = np.stack([klines[s].loc[common, "close"].to_numpy(dtype=np.float64)
                       for s in SYMBOLS], axis=1)

    # apply gap mask (drops both the gap-affected tails and warmup simultaneously
    # — the first WARMUP_BARS rows also fail the mask since they have no prior
    # history of length lookback).
    feats_gs  = feats[gap_safe]
    opens_gs  = opens[gap_safe]
    closes_gs = closes[gap_safe]
    idx_gs    = common[gap_safe]

    # safety net: drop any remaining NaN rows (should be 0 after the warm-up mask)
    nan_row = np.isnan(feats_gs).any(axis=(1, 2))
    dropped_nan = int(nan_row.sum())
    feats_gs  = feats_gs[~nan_row]
    opens_gs  = opens_gs[~nan_row]
    closes_gs = closes_gs[~nan_row]
    idx_gs    = idx_gs[~nan_row]

    # final invariants
    assert not np.isnan(feats_gs).any(),   "NaN in features after cleaning"
    assert not np.isnan(opens_gs).any(),   "NaN in open prices"
    assert not np.isnan(closes_gs).any(),  "NaN in close prices"
    assert feats_gs.shape[1:] == (len(SYMBOLS), F)
    assert opens_gs.shape  == (feats_gs.shape[0], len(SYMBOLS))
    assert closes_gs.shape == (feats_gs.shape[0], len(SYMBOLS))

    # Normalize to ns-since-epoch (int64); source index may be [ms] in pandas 3.x.
    ts_naive = idx_gs.tz_convert("UTC").tz_localize(None)
    ts_ns = np.asarray(ts_naive, dtype="datetime64[ns]").view(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        features=feats_gs,
        open=opens_gs,
        close=closes_gs,
        timestamps=ts_ns,
        feature_names=np.array(FEATURE_NAMES),
        symbols=np.array(SYMBOLS),
    )

    return BuildSummary(
        n_rows=feats_gs.shape[0],
        n_symbols=len(SYMBOLS),
        n_features=F,
        start=idx_gs[0],
        end=idx_gs[-1],
        dropped_warmup=0,  # folded into dropped_gaps now
        dropped_gaps=n_all - feats_gs.shape[0] - dropped_nan,
        dropped_nan_rows=dropped_nan,
    )
