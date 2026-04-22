"""Multi-asset RL trader on Binance USDT-M futures."""

from __future__ import annotations

import torch

__version__ = "0.1.0"

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT")


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device spec to a concrete torch.device.

    "auto" (or None) picks cuda when available, else cpu. Anything else is
    passed through to torch.device(...). Used everywhere a policy forward
    runs so CUDA is the default whenever the hardware is there.
    """
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def maybe_compile_forward(
    net: torch.nn.Module,
    device: torch.device,
    *,
    enable: bool = True,
) -> bool:
    """Monkey-patch ``net.forward = torch.compile(net.forward)`` when enabled
    and on CUDA; return whether compile was applied.

    Gated on CUDA because CPU compile warmup (tens of seconds) typically
    dwarfs any speedup on short runs. Patching ``.forward`` on the instance
    (rather than wrapping the module) ensures ``act()`` and ``evaluate()`` —
    both of which call ``self.forward(...)`` directly — also route through
    the compiled graph.
    """
    if enable and device.type == "cuda":
        net.forward = torch.compile(net.forward)
        return True
    return False
