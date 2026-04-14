# market_intelligence/src/scenarios.py
#
# Scenario Analysis Layer
#
# The baseline scoring reflects 2023–2024 market conditions with equal weighting
# of demand and energy attractiveness. Three alternative scenarios test how the
# ranking would shift if key structural conditions changed.
#
# Scenarios are implemented in two ways:
#   1. Pillar weight adjustments — changing what matters most (e.g., feasibility
#      becomes more important when grids are constrained)
#   2. Metric value adjustments — changing the underlying inputs (e.g., power
#      prices increase uniformly, or interconnection queues lengthen)
#
# Both types of adjustment are transparent and documented per scenario.
# "Most robust" markets are those that maintain a top-half ranking across
# all scenarios — these are the markets with the broadest structural advantages.

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import pandas as pd

from market_intelligence.src.processing import normalize_all
from market_intelligence.src.scoring import score_markets


# ── Scenario definitions ──────────────────────────────────────────────────────
#
# Each scenario defines:
#   label               — short display name
#   description         — what structural change this scenario represents
#   pillar_weight_overrides — replace specific pillar weights (others unchanged)
#   metric_multipliers  — multiply specific raw metric values by a factor
#                         before re-normalizing (applied to all markets equally)
#   narrative           — one paragraph explaining what this scenario implies
#                         for the ranking and why

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "label":       "Baseline (2023–2024 conditions)",
        "description": "Current market conditions. Demand and energy weighted equally at 30% each.",
        "pillar_weight_overrides": {},
        "metric_multipliers":      {},
        "narrative": (
            "The baseline scenario reflects approximate 2023–2024 conditions across all "
            "eight markets. Demand and energy attractiveness are weighted equally, reflecting "
            "the view that a project requires both paying customers and viable power supply "
            "to be commercially feasible. Pacific Northwest and ERCOT Texas lead."
        ),
    },

    "high_ai_demand": {
        "label":       "High AI Demand Growth",
        "description": (
            "AI infrastructure demand accelerates significantly. Data center capacity "
            "pipeline grows 40% across markets; demand pillar weight increases. "
            "Reflects a scenario where hyperscaler capex accelerates and new markets "
            "begin receiving serious tenant interest."
        ),
        "pillar_weight_overrides": {
            "demand":      0.40,   # up from 0.30 — demand scarcity drives prioritization
            "energy":      0.25,   # down from 0.30
            "feasibility": 0.25,   # unchanged
            "strategic":   0.10,   # down from 0.15
        },
        "metric_multipliers": {
            # Existing capacity becomes more valuable; multiply to simulate growth
            "data_center_capacity_mw": 1.40,
            # Hyperscale presence scores increase modestly in active markets
            # (bounded at 5, so this has asymmetric effect on lower-scored markets)
            "hyperscale_presence":     1.15,
        },
        "narrative": (
            "Under rapid AI demand growth, markets with established infrastructure and "
            "hyperscale presence pull further ahead because new entrants follow existing "
            "demand concentration. Northern Virginia becomes more competitive despite its "
            "feasibility constraints. Iowa and Kansas remain penalized by low baseline demand "
            "even as the overall market grows — they benefit from higher demand scores "
            "only if an anchor tenant commits. ERCOT Texas benefits strongly from its "
            "combination of growing demand and best-in-class execution environment."
        ),
    },

    "rising_power_prices": {
        "label":       "Rising Power Prices (+35%)",
        "description": (
            "Wholesale electricity prices increase 35% across all markets due to "
            "sustained high natural gas prices, increased grid demand from electrification, "
            "and capacity tightening. The absolute spread between cheap and expensive "
            "markets widens, making energy cost more differentiating."
        ),
        "pillar_weight_overrides": {
            "demand":      0.25,   # down from 0.30
            "energy":      0.40,   # up from 0.30 — energy cost now more critical
            "feasibility": 0.25,   # unchanged
            "strategic":   0.10,   # down from 0.15
        },
        "metric_multipliers": {
            "avg_wholesale_power_price_mwh": 1.35,
        },
        "narrative": (
            "A 35% uniform increase in wholesale power prices compresses margins in "
            "expensive markets (CAISO, PJM) more than in cheap markets (SPP, MISO) "
            "because the absolute dollar difference widens. At $78/MWh in California "
            "vs. $35/MWh in Iowa, the power cost difference for a 100 MW facility is "
            "approximately $37M per year — a number large enough to drive location "
            "decisions independently of other factors. Iowa and Kansas improve in the "
            "ranking; California falls further. The Pacific Northwest, with hydro-based "
            "power prices that are less correlated with gas prices, holds its position."
        ),
    },

    "tighter_grid_constraints": {
        "label":       "Tighter Grid Constraints",
        "description": (
            "Interconnection queues lengthen 50% across all markets as the volume of "
            "new generation and load applications exceeds ISO/RTO processing capacity. "
            "Permitting timelines also extend modestly. Feasibility becomes a more "
            "decisive factor in market selection."
        ),
        "pillar_weight_overrides": {
            "demand":      0.25,   # down from 0.30
            "energy":      0.25,   # down from 0.30
            "feasibility": 0.40,   # up from 0.25 — execution risk becomes dominant
            "strategic":   0.10,   # down from 0.15
        },
        "metric_multipliers": {
            "interconnection_queue_months": 1.50,
        },
        "narrative": (
            "When interconnection queues lengthen uniformly, markets that were already "
            "fast (ERCOT at 14 months × 1.5 = 21 months) remain materially better than "
            "markets that were already slow (CAISO at 48 months × 1.5 = 72 months). "
            "The ranking impact is asymmetric: fast markets improve their relative "
            "position because they have room to absorb additional delay without "
            "crossing the threshold where a project's timeline becomes commercially "
            "unacceptable. ERCOT and SPP benefit significantly. PJM Northern Virginia "
            "and CAISO become increasingly difficult to justify for greenfield development."
        ),
    },
}


# ── Scenario runner ───────────────────────────────────────────────────────────

def run_scenario(
    raw_df: pd.DataFrame,
    scenario_id: str,
) -> pd.DataFrame:
    """
    Run the full scoring pipeline under a specific scenario's conditions.

    Applies metric multipliers to raw input values, re-normalizes, and
    re-scores with scenario pillar weights. Returns a scored DataFrame
    in the same format as the baseline score_markets() output.

    Bounds are scaled proportionally with metric multipliers to prevent
    clipping artifacts (e.g., a 35% power price increase would otherwise
    push CAISO's $58/MWh × 1.35 = $78.30 above the $65 normalization
    ceiling, causing it to score identically to any market at $65).
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )

    scenario = SCENARIOS[scenario_id]

    # Apply metric multipliers to raw values before normalization.
    # We work on a copy so the original DataFrame is not modified.
    adjusted_df = raw_df.copy()
    for metric, multiplier in scenario["metric_multipliers"].items():
        if metric in adjusted_df.columns:
            adjusted_df[metric] = adjusted_df[metric] * multiplier

    import market_intelligence.src.config as cfg
    original_weights = cfg.PILLAR_WEIGHTS.copy()

    # Scale normalization bounds proportionally with each metric multiplier.
    # This preserves the relative position of each market within the scaled
    # range rather than clipping high-multiplier values at the original ceiling.
    original_bounds: dict[str, tuple] = {}
    for metric, multiplier in scenario["metric_multipliers"].items():
        if metric in cfg.METRICS and cfg.METRICS[metric]["bounds"] is not None:
            low, high = cfg.METRICS[metric]["bounds"]
            original_bounds[metric] = (low, high)
            cfg.METRICS[metric]["bounds"] = (low * multiplier, high * multiplier)

    # Apply pillar weight overrides
    if scenario["pillar_weight_overrides"]:
        merged = {**original_weights, **scenario["pillar_weight_overrides"]}
        cfg.PILLAR_WEIGHTS = merged

    try:
        # Re-normalize with adjusted values and scaled bounds
        metric_scores = normalize_all(adjusted_df)
        scored = score_markets(adjusted_df, metric_scores)
    finally:
        # Always restore original config state
        cfg.PILLAR_WEIGHTS = original_weights
        for metric, bounds in original_bounds.items():
            cfg.METRICS[metric]["bounds"] = bounds

    return scored


def run_all_scenarios(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all defined scenarios and return results as a dict of DataFrames.
    Keys are scenario_ids; values are the full scored DataFrames.
    """
    return {sid: run_scenario(raw_df, sid) for sid in SCENARIOS}


def scenario_rank_comparison(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a side-by-side comparison of market ranks across all scenarios.

    Columns: market_name, baseline_rank, high_ai_demand_rank,
             rising_power_prices_rank, tighter_grid_constraints_rank,
             rank_range (max - min rank across all scenarios).

    rank_range = 0 means a market holds the same rank in every scenario.
    Large rank_range signals a market that is highly sensitive to which
    structural assumption turns out to be correct.
    """
    all_results = run_all_scenarios(raw_df)

    # Start with market identifiers from baseline
    baseline = all_results["baseline"][["market_id", "market_name"]].copy()
    result = baseline.copy()

    rank_cols = []
    for scenario_id, scored in all_results.items():
        col = f"{scenario_id}_rank"
        ranks = scored[["market_id", "rank"]].rename(columns={"rank": col})
        result = result.merge(ranks, on="market_id", how="left")
        rank_cols.append(col)

    # Rank volatility: how much does a market's rank move across scenarios?
    result["rank_range"] = result[rank_cols].max(axis=1) - result[rank_cols].min(axis=1)

    return result.sort_values("baseline_rank").reset_index(drop=True)


def most_robust_markets(raw_df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """
    Identify markets that rank in the top half across all scenarios.

    "Robust" means: regardless of which of the three structural changes
    materializes, the market remains in the upper portion of the ranking.
    These are the markets where a BD team can be most confident that their
    diligence time is well spent.

    Returns a DataFrame of qualifying markets sorted by average rank.
    """
    comparison = scenario_rank_comparison(raw_df)
    scenario_rank_cols = [c for c in comparison.columns
                          if c.endswith("_rank") and c != "rank_range"]

    n_markets = len(comparison)
    top_half_threshold = n_markets / 2

    # A market is "robust" if its worst rank across all scenarios is still
    # in the top half.
    comparison["worst_rank"] = comparison[scenario_rank_cols].max(axis=1)
    comparison["avg_rank"]   = comparison[scenario_rank_cols].mean(axis=1).round(1)

    robust = (
        comparison[comparison["worst_rank"] <= top_half_threshold]
        .sort_values("avg_rank")
        .reset_index(drop=True)
    )

    return robust[["market_name", "avg_rank", "worst_rank", "rank_range"]
                  + scenario_rank_cols]
