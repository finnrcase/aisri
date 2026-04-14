# market_intelligence/src/processing.py
#
# Responsible for two things only:
#   1. Loading and validating the raw market input dataset
#   2. Normalizing each metric to a 0–100 score
#
# Normalization uses min-max scaling bounded by the explicit ranges defined in
# config.py. Direction is handled so that "lower is better" metrics (e.g.,
# power price, queue wait time) are inverted — a lower raw value maps to a
# higher score.
#
# Output: a DataFrame of metric scores (0–100), one column per metric,
# one row per market. The market identifiers are preserved as the index.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from market_intelligence.src.config import METRICS

# ── Column categories ────────────────────────────────────────────────────────

# These columns identify a market but are never scored.
ID_COLS = ["market_id", "market_name", "iso_rto", "primary_states"]

# Free-text annotation column — carried through but not used in scoring.
NOTE_COL = "notes"

# The complete set of metric columns the pipeline expects to find in the CSV.
METRIC_COLS = list(METRICS.keys())


# ── Data loading ─────────────────────────────────────────────────────────────

def load_markets(path: str | Path) -> pd.DataFrame:
    """
    Load market_inputs.csv and perform basic structural validation.

    Returns a DataFrame with all ID columns and metric columns present.
    Raises ValueError with a descriptive message if any expected columns
    are missing — catches data entry errors before they cause silent NaNs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Market input file not found: {path}")

    df = pd.read_csv(path)

    # Confirm all expected metric columns are present.
    # This catches typos in the CSV header and config mismatches early.
    missing = [col for col in METRIC_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"market_inputs.csv is missing expected metric columns: {missing}\n"
            f"Check that the column names match the keys in config.METRICS."
        )

    # Confirm all identity columns are present.
    missing_id = [col for col in ID_COLS if col not in df.columns]
    if missing_id:
        raise ValueError(f"market_inputs.csv is missing identity columns: {missing_id}")

    # Coerce all metric columns to numeric. Non-numeric values become NaN,
    # which will be caught downstream rather than silently skipped here.
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_nulls = df[METRIC_COLS].isnull().sum().sum()
    if n_nulls > 0:
        bad = df[METRIC_COLS].isnull().stack()
        bad = bad[bad].index.tolist()
        print(f"[WARN] {n_nulls} null value(s) found in metric columns: {bad}")
        print("       Null metrics will receive a score of 0 (most conservative assumption).")

    return df


# ── Normalization ─────────────────────────────────────────────────────────────

def _resolve_bounds(
    series: pd.Series,
    config_bounds: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Return the (low, high) bounds to use for normalization.

    If explicit bounds are defined in config.py, use those. This keeps scores
    stable when new markets are added — a new market at the top of the range
    won't rescale all existing markets.

    If bounds are None in config (not currently the case), fall back to the
    observed min/max in the dataset.
    """
    if config_bounds is not None:
        return config_bounds

    # Dynamic fallback: derive from the dataset.
    low = float(series.min())
    high = float(series.max())

    if abs(high - low) < 1e-9:
        # All markets have the same value — score everyone at 50 (neutral).
        return (low - 1.0, high + 1.0)

    return (low, high)


def normalize_metric(
    series: pd.Series,
    bounds: Optional[Tuple[float, float]],
    direction: str,
) -> pd.Series:
    """
    Normalize a single metric column to a 0–100 opportunity score.

    For "higher_is_better" metrics (e.g., renewable capacity factor):
        score = (value - low) / (high - low) * 100
        → the maximum raw value maps to 100 (best opportunity)

    For "lower_is_better" metrics (e.g., power price, queue wait time):
        score = (high - value) / (high - low) * 100
        → the minimum raw value maps to 100 (best opportunity)

    Values outside the defined bounds are clipped before scoring, so a market
    that reports a capacity factor above the config ceiling still scores 100
    rather than above it.
    """
    low, high = _resolve_bounds(series, bounds)
    span = high - low

    # Replace NaN with the worst possible score (0) — conservative assumption.
    # A missing value is treated as no data = no advantage.
    values = series.fillna(low if direction == "higher_is_better" else high)

    # Clip to the defined range so out-of-range values don't break scaling.
    clipped = values.clip(lower=low, upper=high)

    if direction == "higher_is_better":
        normalized = (clipped - low) / span
    else:  # lower_is_better — invert so that cheaper/faster = higher score
        normalized = (high - clipped) / span

    return (normalized * 100).round(2)


def normalize_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalize_metric to every metric column defined in config.METRICS.

    Returns a new DataFrame (index = market_id) where each column contains
    the 0–100 opportunity score for that metric. The raw input values are
    not retained here — the caller combines them with the scored table.
    """
    df = df.set_index("market_id")
    scored = pd.DataFrame(index=df.index)

    for metric_id, cfg in METRICS.items():
        raw_series = df[metric_id]
        score_series = normalize_metric(
            series=raw_series,
            bounds=cfg["bounds"],
            direction=cfg["direction"],
        )
        # Name the output column with a _score suffix to distinguish it from
        # the raw input value in any combined table.
        scored[f"{metric_id}_score"] = score_series

    return scored
