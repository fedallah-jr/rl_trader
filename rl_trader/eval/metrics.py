"""Summary statistics over a RolloutResult."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rollout import RolloutResult


@dataclass
class EvalMetrics:
    n_steps: int
    total_pnl: float
    total_fees: float
    net_pnl: float
    ann_sharpe: float
    max_drawdown: float
    hit_rate: float                       # fraction of in-position steps with net > 0
    turnover_per_day: float               # avg #leg-changes per calendar day
    time_flat_frac: float                 # fraction of (step, symbol) pairs fully flat
    per_symbol_turnover: dict[str, int]


def compute_metrics(res: RolloutResult, notional_U: float = 1.0) -> EvalMetrics:
    r    = res.rewards
    pnl  = res.step_pnl
    fees = res.fees
    net  = pnl - fees

    total_pnl = float(pnl.sum())
    total_fees = float(fees.sum())
    net_pnl = float(net.sum())

    if r.std() > 0:
        ann_sharpe = float(r.mean() / r.std() * np.sqrt(60 * 24 * 365))
    else:
        ann_sharpe = 0.0

    cum = res.cum_pnl
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_drawdown = float(dd.min() / notional_U)

    in_pos_any = (res.positions != 0).any(axis=1)
    hit_rate = float((net[in_pos_any] > 0).mean()) if in_pos_any.any() else float("nan")

    turnover_per_symbol: dict[str, int] = {}
    for i, s in enumerate(res.symbols):
        pos_i = res.positions[:, i]
        changes = int((pos_i[1:] != pos_i[:-1]).sum()) + int(pos_i[0] != 0)
        turnover_per_symbol[s] = changes

    minutes = int(res.timestamps.size)
    days = max(1.0, minutes / (60 * 24))
    turnover_per_day = sum(turnover_per_symbol.values()) / days
    time_flat_frac = float((res.positions == 0).mean())

    return EvalMetrics(
        n_steps=minutes,
        total_pnl=total_pnl,
        total_fees=total_fees,
        net_pnl=net_pnl,
        ann_sharpe=ann_sharpe,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        turnover_per_day=turnover_per_day,
        time_flat_frac=time_flat_frac,
        per_symbol_turnover=turnover_per_symbol,
    )


def summary_table(m: EvalMetrics) -> str:
    lines = [
        f"  n_steps           : {m.n_steps:,}",
        f"  total pnl         : {m.total_pnl:+.4f}",
        f"  total fees        : {m.total_fees:.4f}",
        f"  net pnl           : {m.net_pnl:+.4f}",
        f"  annualised Sharpe : {m.ann_sharpe:+.3f}",
        f"  max drawdown      : {m.max_drawdown:+.4f}",
        f"  hit rate          : {m.hit_rate:.3f}",
        f"  turnover / day    : {m.turnover_per_day:.2f}",
        f"  time flat (frac)  : {m.time_flat_frac:.3f}",
        f"  turnover by symbol:",
    ]
    for s, n in m.per_symbol_turnover.items():
        lines.append(f"    {s:<8} {n:>6,}")
    return "\n".join(lines)
