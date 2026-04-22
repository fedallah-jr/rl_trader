# Binance USDT-M Futures Dataset

Historical 1-minute OHLCV klines and 5-minute futures metrics for six
USDT-margined perpetuals, sourced from
[data.binance.vision](https://data.binance.vision/).

## Layout

```
dataset/
├── raw/                                           original zips + CHECKSUM-verified
│   ├── klines/{SYMBOL}/1m/{SYMBOL}-1m-YYYY-MM.zip
│   └── metrics/{SYMBOL}/{SYMBOL}-metrics-YYYY-MM-DD.zip
└── processed/                                     per-symbol pandas pickles
    ├── klines/{SYMBOL}_1m.pkl
    └── metrics/{SYMBOL}_metrics.pkl
```

## Coverage

- **Symbols**: BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, BNBUSDT, XRPUSDT
- **Klines**: 2021-12-01 → 2026-03-31 (1-minute, 52 monthly files per symbol)
- **Metrics**: 2021-12-01 → 2026-04-20 (5-minute, 1602 daily files per symbol)

The start date is the earliest month for which Binance publishes metrics for
all six symbols.

## Loading

```python
import pandas as pd

k = pd.read_pickle("dataset/processed/klines/BTCUSDT_1m.pkl")
m = pd.read_pickle("dataset/processed/metrics/BTCUSDT_metrics.pkl")
```

Both DataFrames have a UTC `DatetimeIndex`, are sorted, de-duplicated, and
have the `ignore` column dropped from klines.

### `{SYMBOL}_1m.pkl` — columns

Index is `open_time` (UTC, left edge of the minute bar).

| Column                   | Dtype              | Meaning                         |
|--------------------------|--------------------|---------------------------------|
| `open`                   | float64            | Open price                      |
| `high`                   | float64            | High price                      |
| `low`                    | float64            | Low price                       |
| `close`                  | float64            | Close price                     |
| `volume`                 | float64            | Base-asset volume               |
| `close_time`             | datetime64[ms, UTC]| Close timestamp (open_time + 59.999s) |
| `quote_volume`           | float64            | Quote-asset (USDT) volume       |
| `count`                  | Int64              | Number of trades                |
| `taker_buy_volume`       | float64            | Taker-buy base volume           |
| `taker_buy_quote_volume` | float64            | Taker-buy quote volume          |

Per symbol: 2,278,080 rows (BTC / ETH / ADA / BNB) or 2,270,880 rows
(SOL / XRP — these two are missing 5 days in 2022-02 and 2022-04 because
Binance Vision's monthly zips for those symbols were short in those months).

### `{SYMBOL}_metrics.pkl` — columns

Index is `create_time` (UTC, snapshot time).

| Column                             | Dtype    | Meaning                                 |
|------------------------------------|----------|-----------------------------------------|
| `sum_open_interest`                | float64  | Total open interest (base asset)        |
| `sum_open_interest_value`          | float64  | Total open interest (quote asset)       |
| `count_toptrader_long_short_ratio` | float64  | Top-trader long/short *account* ratio   |
| `sum_toptrader_long_short_ratio`   | float64  | Top-trader long/short *position* ratio  |
| `count_long_short_ratio`           | float64  | All-accounts long/short ratio           |
| `sum_taker_long_short_vol_ratio`   | float64  | Taker long/short *volume* ratio         |

Per symbol: ~461,200 rows. Notes on missingness:
- `sum_toptrader_long_short_ratio` and `sum_taker_long_short_vol_ratio` have
  long stretches of `NaN` in 2022 (months at a time) — Binance was not
  computing these for some symbols/periods.
- `sum_open_interest` has ~500 rows where it equals exactly `0` (source
  anomaly, not missing data per se).
- Around 2-3k rows per symbol are stamped at non-5-minute boundaries
  (recovery artifacts after exchange outages).

## Regenerating from the raw zips

Each zip contains a single CSV with Binance Vision's standard format. Raw
klines CSV columns (as documented by Binance):

```
open_time,open,high,low,close,volume,close_time,quote_volume,count,
taker_buy_volume,taker_buy_quote_volume,ignore
```

`open_time` and `close_time` are epoch milliseconds. Files from late 2022
onward have a header row; earlier files do not.

Raw metrics CSV columns:

```
create_time,symbol,sum_open_interest,sum_open_interest_value,
count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,
count_long_short_ratio,sum_taker_long_short_vol_ratio
```

`create_time` is a `"YYYY-MM-DD HH:MM:SS"` string in UTC. Metrics files
always have a header row.

Each `.zip` has a sibling `.CHECKSUM` file holding its SHA256 in the
standard `sha256sum` format.

## Source

- Klines: `https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/1m/`
- Metrics: `https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/`
