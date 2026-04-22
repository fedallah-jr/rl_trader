"""Run-level configs composed from per-module dataclasses.

`RunConfig` is the single source of truth for a training/eval run. It wraps
the three existing configs — `PPOConfig`, `PolicyConfig`, `EnvConfig` — and
provides `from_cli_args()` that builds it from dotted CLI flags like
`--env.window 256  --ppo.lr 1e-4  --architecture.d_model 128`.

Only common primitive types (int, float, str, bool, Optional[str]) are
overridable via CLI. Override other fields by constructing the config in code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from .algorithms.ppo import PPOConfig
from .architectures.factored_attention import PolicyConfig
from .envs.multi_asset import EnvConfig


@dataclass
class SplitsConfig:
    """Date-range splits of the feature dataset.

    Last month -> test (never touched by training loop).
    Previous 3 months -> val (used for periodic eval and best-model selection).
    Everything before that -> train.
    """
    train_start: str = "2021-12-02"
    train_end:   str = "2025-11-30"
    val_start:   str = "2025-12-01"
    val_end:     str = "2026-02-28"
    test_start:  str = "2026-03-01"
    test_end:    str = "2026-03-31"


@dataclass
class RunConfig:
    """Top-level composition of all sub-configs for one run."""

    seed: int = 0
    ppo: PPOConfig = field(default_factory=PPOConfig)
    architecture: PolicyConfig = field(default_factory=PolicyConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)

    # ------------------------------------------------------------------

    @classmethod
    def from_cli_args(cls, argv: list[str] | None = None) -> "RunConfig":
        """Build a RunConfig from CLI flags, using dotted names for nesting.

        Usage:
            RunConfig.from_cli_args(["--env.window", "256", "--ppo.lr", "1e-4"])

        Fields are looked up recursively. Default values come from each
        dataclass's own defaults; CLI flags override only the fields provided.
        """
        cfg = cls()
        parser = argparse.ArgumentParser(
            description="rl_trader run config (nested dataclasses)",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _add_dataclass_args(parser, cfg, prefix="")
        args = parser.parse_args(argv)
        _apply_args(cfg, args, prefix="")
        return cfg

    def pretty(self) -> str:
        """Human-readable recursive dump."""
        return _pretty(self, indent=0)


# ---- internal helpers ----------------------------------------------------

_PRIMITIVES = (int, float, str, bool)


def _unwrap_optional(typ: Any) -> Any:
    """If typ is Optional[X] / Union[X, None], return X; else return typ."""
    origin = get_origin(typ)
    if origin is None:
        return typ
    args = get_args(typ)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    return typ


def _primitive_for(typ: Any, current_value: Any) -> type | None:
    """Resolve a primitive Python type usable with argparse.

    Falls back to the runtime type of the current value when the annotation is
    not resolvable (e.g. string annotations under `from __future__ import
    annotations`).
    """
    typ = _unwrap_optional(typ)
    if isinstance(typ, type) and typ in _PRIMITIVES:
        return typ
    if isinstance(current_value, _PRIMITIVES):
        return type(current_value)
    return None


def _add_dataclass_args(parser: argparse.ArgumentParser, obj: Any, prefix: str) -> None:
    """Walk a dataclass instance, register a CLI flag for each primitive field."""
    try:
        hints = get_type_hints(type(obj))
    except Exception:
        hints = {}
    for f in fields(obj):
        name = f"{prefix}.{f.name}" if prefix else f.name
        value = getattr(obj, f.name)
        if is_dataclass(value):
            _add_dataclass_args(parser, value, prefix=name)
            continue
        prim = _primitive_for(hints.get(f.name), value)
        if prim is None:
            continue                       # skip complex fields (lists, dicts, ...)
        flag_underscore = f"--{name}"
        flag_dash       = f"--{name.replace('_', '-')}"
        flags = [flag_dash] if flag_dash == flag_underscore else [flag_dash, flag_underscore]
        if prim is bool:
            parser.add_argument(
                *flags,
                dest=name,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                *flags,
                dest=name,
                type=prim,
                default=argparse.SUPPRESS,
            )


def _apply_args(obj: Any, args: argparse.Namespace, prefix: str) -> None:
    for f in fields(obj):
        name = f"{prefix}.{f.name}" if prefix else f.name
        value = getattr(obj, f.name)
        if is_dataclass(value):
            _apply_args(value, args, prefix=name)
            continue
        if hasattr(args, name):
            setattr(obj, f.name, getattr(args, name))


def _pretty(obj: Any, indent: int) -> str:
    pad = "  " * indent
    if is_dataclass(obj):
        lines = [f"{type(obj).__name__}:"]
        for f in fields(obj):
            v = getattr(obj, f.name)
            if is_dataclass(v):
                lines.append(f"{pad}  {f.name}: {_pretty(v, indent + 1)}")
            else:
                lines.append(f"{pad}  {f.name}: {v!r}")
        return "\n".join(lines)
    return repr(obj)
