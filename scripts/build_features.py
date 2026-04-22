"""CLI: build the feature tensor and write it to dataset/features/features.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_trader.features import FEATURE_NAMES, build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed-root",
        default="dataset/processed",
        help="path with klines/ and metrics/ subdirs of per-symbol pickles",
    )
    ap.add_argument(
        "--out",
        default="dataset/features/features.npz",
        help="output .npz path",
    )
    args = ap.parse_args()

    summary = build_dataset(Path(args.processed_root), Path(args.out))

    print(f"wrote {args.out}")
    print(f"  rows     : {summary.n_rows:,}")
    print(f"  symbols  : {summary.n_symbols}")
    print(f"  features : {summary.n_features}   ({', '.join(FEATURE_NAMES)})")
    print(f"  coverage : {summary.start}  →  {summary.end}")
    print(f"  dropped  : gaps+warmup={summary.dropped_gaps:,}  "
          f"nan_rows={summary.dropped_nan_rows:,}")


if __name__ == "__main__":
    main()
