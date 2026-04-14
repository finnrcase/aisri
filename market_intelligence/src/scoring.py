# market_intelligence/src/scoring.py
#
# Takes the normalized metric scores produced by processing.py and aggregates
# them into pillar scores and a final composite opportunity score.
#
# Aggregation is a two-step weighted average:
#   Step 1: metric scores → pillar scores  (weights defined per pillar in config)
#   Step 2: pillar scores → composite score (pillar weights defined in config)
#
# The composite opportunity_score ranges 0–100. Higher = better market.
# Markets are ranked from highest to lowest opportunity.

from __future__ import annotations

import pandas as pd

from market_intelligence.src.config import METRICS, PILLAR_METRICS, PILLAR_WEIGHTS


# ── Pillar scoring ────────────────────────────────────────────────────────────

def compute_pillar_scores(metric_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate normalized metric scores into one score per pillar per market.

    For each pillar, this is a weighted average of its constituent metric scores,
    using the per-metric weights defined in config.METRICS.

    Input:  DataFrame indexed by market_id, columns = "{metric_id}_score"
    Output: DataFrame indexed by market_id, columns = "{pillar}_score"

    Why weighted average rather than simple average?
    Within the energy pillar, wholesale power price (35% weight) matters more
    than grid reliability (15% weight) — a weighted average lets the config
    encode those relative judgements explicitly and traceably.
    """
    pillar_scores = pd.DataFrame(index=metric_scores.index)

    for pillar, metric_ids in PILLAR_METRICS.items():
        # Retrieve the weight for each metric within this pillar.
        weights = [METRICS[m]["weight"] for m in metric_ids]

        # Stack the relevant metric score columns into a sub-DataFrame.
        score_cols = [f"{m}_score" for m in metric_ids]
        sub = metric_scores[score_cols]

        # Weighted average: sum(score_i * weight_i) / sum(weight_i)
        # Using matrix multiplication for clarity and efficiency.
        weight_array = pd.Series(weights, index=score_cols)
        pillar_scores[f"{pillar}_score"] = sub.dot(weight_array).round(2)

    return pillar_scores


# ── Composite scoring ─────────────────────────────────────────────────────────

def compute_composite_score(pillar_scores: pd.DataFrame) -> pd.Series:
    """
    Aggregate pillar scores into a single composite opportunity score per market.

    This is the final weighted average across all four pillars, using the
    top-level pillar weights from config.PILLAR_WEIGHTS.

    Returns a Series indexed by market_id named "opportunity_score".

    The two-level aggregation (metric → pillar → composite) means the overall
    score reflects the analytical structure rather than collapsing 13 metrics
    into a flat average. A pillar with more metrics doesn't automatically
    dominate — each pillar's influence is controlled by its pillar weight.
    """
    pillar_cols = [f"{p}_score" for p in PILLAR_WEIGHTS.keys()]
    weights = pd.Series(
        list(PILLAR_WEIGHTS.values()),
        index=pillar_cols,
    )

    composite = pillar_scores[pillar_cols].dot(weights).round(2)
    composite.name = "opportunity_score"
    return composite


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_markets(composite_scores: pd.Series) -> pd.Series:
    """
    Assign integer ranks 1..N where 1 = highest opportunity score.

    Ties receive the same rank (method="min") so the highest-ranked market
    always has rank 1, regardless of tie-breaking behaviour.
    """
    return composite_scores.rank(ascending=False, method="min").astype(int).rename("rank")


# ── Final table assembly ──────────────────────────────────────────────────────

def build_scored_table(
    raw_df: pd.DataFrame,
    metric_scores: pd.DataFrame,
    pillar_scores: pd.DataFrame,
    composite: pd.Series,
    ranks: pd.Series,
) -> pd.DataFrame:
    """
    Assemble all scoring outputs into a single clean output table.

    Column order is designed for readability:
      1. Market identifiers (non-scored)
      2. Rank and composite opportunity score  ← the headline numbers
      3. Pillar scores                          ← one level down
      4. Individual metric scores              ← full transparency / audit trail

    Having all three tiers in one CSV means a reviewer can see not just which
    market ranked first, but exactly which metrics drove that result.
    """
    # Identity columns — set market_id as index to align joins
    id_cols = ["market_name", "iso_rto", "primary_states"]
    identifiers = raw_df.set_index("market_id")[id_cols]

    table = (
        identifiers
        .join(ranks)
        .join(composite)
        .join(pillar_scores)
        .join(metric_scores)
    )

    # Sort by rank ascending (rank 1 at the top)
    table = table.sort_values("rank").reset_index()

    return table


# ── Public pipeline entry point ───────────────────────────────────────────────

def score_markets(raw_df: pd.DataFrame, metric_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full scoring pipeline given raw market data and normalized metric scores.

    This is the function called by run_market_score.py.
    It is intentionally a thin orchestrator — each step is independently testable.

    Steps:
        1. Aggregate metric scores → pillar scores
        2. Aggregate pillar scores → composite opportunity score
        3. Rank markets by composite score
        4. Assemble final output table
    """
    pillar_scores = compute_pillar_scores(metric_scores)
    composite = compute_composite_score(pillar_scores)
    ranks = rank_markets(composite)
    return build_scored_table(raw_df, metric_scores, pillar_scores, composite, ranks)
