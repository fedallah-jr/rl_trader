"""CLI: evaluate a trained PPO checkpoint on a date range."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl_trader.envs import EnvConfig, MultiAssetTradingEnv
from rl_trader.eval import compute_metrics, load_policy, run_rollout, summary_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to .pt file")
    ap.add_argument("--features", default="dataset/features/features.npz")
    ap.add_argument("--start-ts", default="2025-07-01")
    ap.add_argument("--end-ts",   default="2025-12-31")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=None, help="cap on rollout length")
    ap.add_argument("--start-idx", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", default=None, help="optional path to .npz dump of rollout")
    args = ap.parse_args()

    env = MultiAssetTradingEnv(EnvConfig(
        features_path=args.features,
        window=args.window,
        episode_min=1,
        episode_max=args.n_steps or 10_000_000,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    ))
    net = load_policy(args.checkpoint, device=args.device)

    print(f"evaluating {args.checkpoint} on {args.start_ts} → {args.end_ts}")
    res = run_rollout(
        env, net,
        start_idx=args.start_idx,
        n_steps=args.n_steps,
        deterministic=not args.stochastic,
        device=args.device,
    )
    m = compute_metrics(res, notional_U=env.cfg.notional_U)
    print(summary_table(m))

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            timestamps=res.timestamps,
            actions=res.actions,
            positions=res.positions,
            rewards=res.rewards,
            step_pnl=res.step_pnl,
            fees=res.fees,
            cum_pnl=res.cum_pnl,
            symbols=np.array(res.symbols),
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
