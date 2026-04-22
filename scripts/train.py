"""CLI: train the PPO agent with periodic validation + best-model tracking.

Splits (override via `--splits.*`):
    train : 2021-12-02 → 2025-11-30   ~3.8 years
    val   : 2025-12-01 → 2026-02-28   3 months (periodic eval during training)
    test  : 2026-03-01 → 2026-03-31   1 month (never seen during training)

Every `--ppo.eval-interval-steps` env steps (default 50 000), the agent is
evaluated on 4 fixed one-week deterministic rollouts inside the val range.
The best-so-far weights (by mean net-PnL) are saved to `<ckpt-dir>/best.pt`
whenever they improve, and restored into the returned network at the end.

Examples:
    python -m scripts.train
    python -m scripts.train --ppo.total-timesteps 2000000 --env.window 256
    python -m scripts.train --ppo.eval-interval-steps 20000 --ppo.eval-episodes 6
"""

from __future__ import annotations

from dataclasses import replace

from rl_trader.algorithms.ppo import train_ppo
from rl_trader.configs import RunConfig


def main():
    cfg = RunConfig.from_cli_args()
    print(cfg.pretty(), flush=True)

    # Derive per-split env configs from the splits block.
    train_env = replace(
        cfg.env,
        start_ts=cfg.splits.train_start,
        end_ts=cfg.splits.train_end,
    )
    val_env = replace(
        cfg.env,
        start_ts=cfg.splits.val_start,
        end_ts=cfg.splits.val_end,
    )

    train_ppo(cfg.ppo, train_env, cfg.architecture, val_env_cfg=val_env)


if __name__ == "__main__":
    main()
