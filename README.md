# Multi-Asset RL Trader

A reinforcement-learning trader for **6 Binance USDT-M perpetual futures**,
operating at **1-minute resolution** and trading all symbols simultaneously.
Built on Gymnasium + PyTorch, trained with PPO.

- **Symbols**: BTCUSDT · ETHUSDT · SOLUSDT · ADAUSDT · BNBUSDT · XRPUSDT
- **Data coverage**: 2021-12-02 → 2026-03-31 (≈2.27M 1-minute bars per symbol,
  inner-joined across all 6)
- **Model**: factored space-time attention actor-critic, 1.3 M parameters
- **Action**: `MultiDiscrete([3]^6)` — per-symbol target position ∈ {−1, 0, +1}

---

## Quick start

```bash
# Install (editable)
pip install -e .

# 1. Build the feature tensor (offline, one-time).
#    Reads dataset/processed/{klines,metrics}/*.pkl, writes dataset/features/features.npz.
python -m scripts.build_features

# 2. Train PPO on the train split, with periodic eval on val.
#    train: 2021-12-02 → 2025-11-30, val: 2025-12-01 → 2026-02-28.
#    The best-on-val checkpoint is saved to checkpoints/ppo/best.pt.
python -m scripts.train

# 3. Evaluate best.pt on the held-out test split (2026-03).
python -m scripts.eval \
    --checkpoint checkpoints/ppo/best.pt \
    --start-ts 2026-03-01 --end-ts 2026-03-31
```

All CLIs understand the nested `RunConfig` via dotted flags — see
[Configuration](#configuration) below.

---

## What it does

**Environment.** At each 1-minute bar the agent observes 128 minutes of
per-symbol features (prices, volumes, open-interest deltas, long/short ratios,
staleness flags) plus its own portfolio state, and emits a signed target position
for each of the 6 symbols. The environment fills at the next bar's open, charges
a 5 bps taker fee on turnover, marks to market using close prices, and emits a
**volatility-scaled reward**:
```
reward = Σ_i (ΔPnL_i − fees_i) · (σ_target / σ_i(t))  /  U
```
where `σ_i(t)` is symbol `i`'s trailing 60-minute realised volatility and
`σ_target` is a scalar (default 10 bps per 1m bar). This equalises each
symbol's gradient contribution regardless of its inherent volatility (so SOL
doesn't dominate BTC during training) — following Zhang, Zohren & Roberts
(arXiv:1911.10107), adapted to keep our fixed-notional target-position
semantics intact. **Positions, fills, fees, and the cumulative PnL reported
on eval/test are unchanged** — only the training reward is reweighted. Set
`EnvConfig.sigma_target = None` to fall back to the raw reward.

Action semantics are **idempotent target-positions**: repeating the same action
costs nothing; flipping costs 2× fees; opening/closing costs 1× fees.

**Policy.** Each of the 6 symbols is a token with a learned identity embedding.
The encoder is a stack of 3 *factored* blocks — each alternates
(a) causal self-attention over the 128-step time axis within each symbol, then
(b) full self-attention across the 6 symbols at each time-step. A learned query
attention-pools the time axis. A shared-weights MLP head then takes
`[z_symbol ; account_state_symbol ; globals]` and emits 3 logits per symbol
(MultiDiscrete). The critic mean-pools over symbols and returns a scalar V(s).

**Why this shape.** Cross-asset information (especially BTC-led microstructure)
is critical for alt-coin decisions; late fusion loses it. Factored attention
keeps per-symbol identity throughout while letting symbols mix at every layer.
The action head's shared weights + per-symbol identity gives parameter
efficiency *and* per-symbol specialization.

**Algorithm.** CleanRL-style single-file PPO: synchronous vec-envs, GAE, clipped
surrogate, entropy bonus, value loss, optional linear LR anneal. Checkpoints
bundle the `PPOConfig / EnvConfig / PolicyConfig` that produced them so
`eval.py` can reconstruct the exact model.

---

## Architecture

![architecture](architecture.png)

See `architecture.py` for the source that generated this figure.

Compact form:

```
market[B, T=128, 6, F=17]
    → Linear(F→D=128) + time_pos_emb + symbol_emb
    → [TimeAttn(causal, per symbol) → CrossAssetAttn(per t)] × 3
    → AttentionPool over T                          z[B, 6, D]

per-symbol head (weights shared across symbols):
    [z_i ; account_i ; globals]  →  MLP  →  3 logits   →   Categorical
                                                             sample a_i ∈ {-1,0,+1}

critic: mean_i(z)  →  concat globals  →  MLP  →  V(s)
```

---

## Dataset

The model trains on 1-minute OHLCV klines plus 5-minute futures metrics
(open interest, long/short ratios) sourced from
[data.binance.vision](https://data.binance.vision/). Full schema, column
documentation and integrity notes live in [`dataset/README.md`](dataset/README.md).

The feature pipeline (`rl_trader/features/pipeline.py`) produces a single
`dataset/features/features.npz` that the env memory-maps at startup:

| key             | shape / dtype                  | contents                                   |
|-----------------|--------------------------------|--------------------------------------------|
| `features`      | `float32  [T, 6, 17]`          | per-symbol observation features            |
| `open`          | `float64  [T, 6]`              | 1-minute open prices for fills             |
| `close`         | `float64  [T, 6]`              | 1-minute close prices for PnL marking      |
| `timestamps`    | `int64    [T]`                 | ns-since-epoch (UTC), 1m cadence           |
| `feature_names` | `str [17]`                     | feature column names                       |
| `symbols`       | `str [6]`                      | symbol order                               |

Key pipeline decisions:
- **Inner join** on the common 1-minute index across all 6 symbols
  (drops SOL/XRP's two missing-day blocks in 2022).
- **Metrics merged with a 3-minute publication-lag shift**, empirically
  verified (Binance publishes the 5m-stamped row 1–2 minutes after the
  boundary). See `experiments/verify_oi_freshness.py`.
- **Warm-up + gap handling**: any row whose 1440-bar history spans a 1m gap
  is dropped.
- **Zero NaN** guaranteed in the output — the three metric columns with
  multi-month gaps are deliberately excluded from the feature set.

---

## Repository layout

```
rl_trader/                              importable package
├── __init__.py                         SYMBOLS tuple
├── configs.py                          RunConfig + from_cli_args (dotted nested CLI)
│
├── features/
│   └── pipeline.py                     offline precompute: klines + metrics → features.npz
│
├── envs/
│   ├── _common.py                      in-process feature cache
│   └── multi_asset.py                  gym.Env: target-position semantics
│
├── architectures/
│   ├── base.py                         AbstractActorCritic interface
│   └── factored_attention.py           the proposed net (1.3M params)
│
├── algorithms/
│   ├── _rollout.py                     SyncVecEnv w/ episode tracking
│   └── ppo.py                          CleanRL-style PPO, train_ppo(ppo, env, policy)
│
└── eval/
    ├── rollout.py                      deterministic rollout + checkpoint loading
    └── metrics.py                      Sharpe, drawdown, turnover, hit-rate

scripts/                                user-facing entrypoints
├── build_features.py                   → dataset/features/features.npz
├── train.py                            → checkpoints/ppo/iter*.pt
└── eval.py                             → summary metrics (+ optional rollout dump)

tests/                                  pytest-free, each file runnable stand-alone
├── test_configs.py
├── test_env.py
├── test_policy.py
├── test_eval.py
└── test_ppo_smoke.py

experiments/                            ad-hoc verification scripts
└── verify_oi_freshness.py              empirical check of metric publication lag

dataset/
├── README.md                           data schema + regeneration notes
├── raw/                                original Binance-vision zips
├── processed/                          per-symbol pickles (kline + metric)
└── features/                           features.npz (built by scripts/build_features.py)

architecture.py / architecture.png      source + rendered model diagram
pyproject.toml                          package metadata
```

---

## Configuration

All runs are parameterised by a single nested dataclass, `rl_trader.configs.RunConfig`:

```python
@dataclass
class RunConfig:
    seed: int = 0
    ppo:          PPOConfig        # rollout / optimisation / PPO-specific
    architecture: PolicyConfig     # network geometry
    env:          EnvConfig        # dataset split, window, fees, notional, …
```

CLIs expose every primitive field as a dotted flag. Both dash- and
underscore-style names are accepted:

```bash
python -m scripts.train \
    --seed 42 \
    --ppo.total-timesteps 2000000 \
    --ppo.lr 2e-4 \
    --ppo.n-envs 16 \
    --env.window 256 \
    --env.start-ts 2022-01-01 \
    --env.end-ts 2025-06-30 \
    --architecture.d-model 256 \
    --architecture.n-layers 4
```

`python -m scripts.train --help` lists every flag.

Construct a config from code when you want non-primitive overrides:

```python
from rl_trader.configs import RunConfig
from rl_trader.envs import EnvConfig

cfg = RunConfig()
cfg.env = EnvConfig(window=256, episode_min=2880, bankruptcy_K=5.0)
```

---

## Train / val / test splits

The pipeline produces ≈ 2,266,560 minute bars. The splits are a first-class
block of `RunConfig`, not manual date flags:

| split  | date range                | approx rows  | role                                             |
|--------|---------------------------|--------------|--------------------------------------------------|
| train  | 2021-12-02 → 2025-11-30   | 2,092,000    | rollouts + gradient updates                      |
| val    | 2025-12-01 → 2026-02-28   |   129,600    | periodic eval during training, best-model pick   |
| test   | 2026-03-01 → 2026-03-31   |    44,640    | never seen during training; held for final eval  |

Defaults live in `rl_trader.configs.SplitsConfig`. Override any boundary via
`--splits.train-start ...` etc.

During training, every `--ppo.eval-interval-steps` env steps (default 50 000),
the agent is evaluated on `--ppo.eval-episodes` deterministic rollouts spaced
evenly across the val range (default: 4 one-week rollouts). The best-so-far
weights (by mean net-PnL across those rollouts) are saved to
`<ckpt-dir>/best.pt` whenever they improve, and the returned network has the
best weights restored when training ends.

---

## Adding new components

The subpackage layout is designed so each axis is extensible independently:

| want to add …        | where                                       | contract                                               |
|----------------------|---------------------------------------------|--------------------------------------------------------|
| a new RL algorithm   | `rl_trader/algorithms/<algo>.py`            | export `train_<algo>(<algo>Config, EnvConfig, PolicyConfig)` |
| a new architecture   | `rl_trader/architectures/<net>.py`          | subclass `AbstractActorCritic`                         |
| a new env family     | `rl_trader/envs/<env>.py`                   | subclass `gym.Env`                                     |
| a new feature set    | extend `rl_trader/features/pipeline.py`     | output must be NaN-free `[T, 6, F]` float32            |
| a new reward         | (deferred) `rl_trader/rewards/<reward>.py`  | TBD — extract once we have ≥ 2 variants                |

The three configs (`PPOConfig / EnvConfig / PolicyConfig`) are passed
separately to `train_*`, so a new algorithm just defines its own config dataclass
alongside existing env/architecture configs.

---

## Design decisions worth flagging

- **Fixed notional, no equity tracking.** Each opened position is a fixed
  USDT-notional `U` (default 1.0 so reward is scale-free). The env does not
  track margin or bankruptcy in the financial sense; instead it terminates with
  a large negative reward if cumulative PnL drops below `−K · U` (default K=3).
- **Decision at close(t), fill at open(t+1).** No lookahead; intra-bar PnL is
  cleanly attributable.
- **3-minute metric shift.** At 1m step `t`, only metric rows with
  `stamp ≤ t − 3 min` are used. Verified empirically that Binance's
  publication lag is ~60–120s.
- **Shared actor weights across symbols + per-symbol embedding.** Better data
  efficiency than 6 independent encoders; the embedding breaks symmetry where
  it matters.
- **Vol-scaled reward** (Zhang et al. 2019 adaptation, option A in our
  discussion). Training reward is per-symbol PnL/fees weighted by
  `σ_target / σ_i(t)`, using the `rvol_60m` column from the precomputed
  features. Positions and realised PnL are untouched; best-model selection
  and eval-time metrics continue to be computed on **raw USDT PnL** so
  deployment numbers stay interpretable. Further shaping (drawdown penalty,
  turnover penalty, Sharpe-targeted) is a single-function swap in
  `env.step()` away.

---

## Tests

Every test file is a standalone script (no pytest needed). Run them with the
project virtualenv active:

```bash
python -m tests.test_configs
python -m tests.test_env
python -m tests.test_policy
python -m tests.test_eval
python -m tests.test_ppo_smoke      # runs 2 PPO iterations end-to-end
```

Coverage highlights: observation/action space shapes, idempotent-hold zero-fee
invariant, flip=2× fee invariant, buy-and-hold closed-form PnL equality,
actor-critic interface conformance, CLI override round-trip, end-to-end PPO
training stability (pg loss, KL, entropy, explained variance).

---

## Status

**Done**
- Full data pipeline (17 per-symbol features, NaN-free, publication-lag-aware).
- Multi-asset Gymnasium env with idempotent target-position semantics.
- Factored space-time attention actor-critic (1.29 M params).
- CleanRL-style PPO trainer with CPU/GPU + checkpointing.
- **Periodic val-set eval during training + best-model tracking** (`best.pt`).
- Deterministic eval harness with Sharpe, drawdown, turnover, per-symbol breakdown.
- Nested dataclass configs with dotted CLI overrides.

**Next**
- Further reward shaping (drawdown penalty, turnover penalty, Sharpe target).
- A real training run + hyperparameter sweep.
- Second algorithm (SAC) to exercise the abstraction layer.
- Async vec-envs once CPU rollout becomes the bottleneck.
- Evaluation plots (equity curves, per-symbol attribution, rolling Sharpe).

---

## Requirements

- Python ≥ 3.12
- `numpy ≥ 2.0`, `pandas ≥ 2.3`, `torch ≥ 2.6`, `gymnasium ≥ 1.2`

Set `OMP_NUM_THREADS=2` (or equivalent) when running on CPU — PyTorch's default
thread pool is oversized for batches this small and hurts throughput otherwise.
The PPO trainer does this automatically via `cfg.ppo.torch_threads` (default 2).
