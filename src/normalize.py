# src/normalize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


Direction = Literal["higher_is_better", "lower_is_better"]


@dataclass(frozen=True)
class Bounds:
    low: float
    high: float


V1_BOUNDS = {
    "renewable_share_pct": Bounds(0.0, 100.0),
    "pue": Bounds(1.10, 1.80),
    "scope2_intensity": Bounds(0.0, 0.80),  # tCO2e / MWh
    "offset_share_scope2": Bounds(0.0, 100.0),
}


def _clip(x: float, low: float, high: float) -> float:
    return float(np.clip(x, low, high))


def linear_risk(x: float, bounds: Bounds, direction: Direction) -> float:
    """
    Returns 0–100 risk score using a bounded linear transform.
    """
    x_c = _clip(x, bounds.low, bounds.high)
    span = bounds.high - bounds.low
    if span <= 0:
        raise ValueError("Invalid bounds span")

    if direction == "lower_is_better":
        return 100.0 * (x_c - bounds.low) / span
    else:
        return 100.0 * (1.0 - (x_c - bounds.low) / span)


def binary_risk(v: int) -> float:
    """
    v=1 (disclosed/yes) => risk 0
    v=0 (no) => risk 100
    """
    if v not in (0, 1):
        raise ValueError("binary value must be 0 or 1")
    return 0.0 if v == 1 else 100.0


def assurance_risk(level: int) -> float:
    """
    0 none => 100
    1 limited => 50
    2 reasonable => 0
    """
    if level == 0:
        return 100.0
    if level == 1:
        return 50.0
    if level == 2:
        return 0.0
    raise ValueError("assurance level must be 0, 1, or 2")
