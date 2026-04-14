# market_intelligence/src/customer_targeting.py
#
# Customer Targeting Layer
#
# The baseline scoring framework weights all four pillars from the perspective
# of a clean energy developer evaluating AI infrastructure markets broadly.
# Different customer types have meaningfully different requirements — a
# hyperscaler optimizing for scale and clean energy commitments cares about
# different variables than a frontier AI lab optimizing for training compute cost.
#
# This module re-runs the pillar aggregation with archetype-specific weights,
# producing a separate opportunity ranking per customer type. The metric-level
# normalization is identical to the baseline — only the pillar weights change.
#
# The result answers: "which market should we prioritize if our target tenant
# is a hyperscaler?" vs. "which market should we prioritize if our target
# tenant is an AI lab?"

from __future__ import annotations

import pandas as pd

from market_intelligence.src.config import PILLAR_WEIGHTS
from market_intelligence.src.scoring import (
    compute_pillar_scores,
    compute_composite_score,
    rank_markets,
)


# ── Archetype definitions ─────────────────────────────────────────────────────
#
# Each archetype defines:
#   label           — display name
#   pillar_weights  — how this customer type weights the four pillars
#                     (must sum to 1.0)
#   description     — what this customer type is and what they care about
#   key_requirements — 3-5 bullet points describing their specific needs
#   market_signals  — what market characteristics make them say yes vs. no
#
# Weight rationale per archetype:
#
#   HYPERSCALER: Demand signal matters most — hyperscalers want to be where
#   the ecosystem already exists (fiber, workforce, regulatory familiarity).
#   Energy is real but they have dedicated procurement teams who can solve
#   it. Feasibility matters because they're building at scale. Strategic
#   fit is less differentiating because they bring their own clean energy teams.
#
#   AI LAB: Energy dominates because training compute is a power-cost business.
#   A frontier AI lab running a $100M training run cares intensely about
#   $/MWh. Demand is less critical — they do not need an existing ecosystem,
#   they create one. Feasibility matters because they need to get to power
#   quickly to stay competitive.
#
#   ENTERPRISE: Demand is the primary filter — enterprise IT needs to be
#   close to its users and within existing IT vendor ecosystems. Energy
#   economics are secondary; power is a real cost but not the majority of
#   an enterprise IT budget the way it is for hyperscale or AI training.
#   Feasibility matters for timeline but is less acute than for a 500 MW campus.

ARCHETYPES: dict[str, dict] = {
    "hyperscaler": {
        "label": "Hyperscaler (AWS / Azure / Google Cloud / Meta)",
        "pillar_weights": {
            "demand":      0.40,
            "energy":      0.25,
            "feasibility": 0.25,
            "strategic":   0.10,
        },
        "description": (
            "Large cloud platform operators building multi-hundred-megawatt campus "
            "infrastructure. Characterized by long-term site commitments (15–20 years), "
            "in-house renewable energy procurement teams, and strict uptime requirements. "
            "Seek markets where ecosystem infrastructure already exists and where they "
            "can execute at scale without being the first mover."
        ),
        "key_requirements": [
            "High fiber density and low latency to major population centers",
            "Proven utility and interconnection infrastructure at scale",
            "Existing tech workforce for operations and engineering",
            "100% clean energy matching (own procurement teams handle this)",
            "Regulatory and permitting environment that supports large-format development",
        ],
        "differentiating_signal": (
            "Existing hyperscale presence in the market is the strongest positive signal "
            "— it confirms that the market has passed the threshold that hyperscalers "
            "require. Northern Virginia is the canonical example: every major operator "
            "is there because every major operator is there."
        ),
    },

    "ai_lab": {
        "label": "Frontier AI Lab (Training-Focused Compute)",
        "pillar_weights": {
            "demand":      0.15,
            "energy":      0.50,
            "feasibility": 0.25,
            "strategic":   0.10,
        },
        "description": (
            "AI research organizations and frontier model developers building large-scale "
            "training infrastructure. Characterized by extremely high power density "
            "(50–100+ MW per cluster), strong cost sensitivity on power (training runs "
            "are measured in megawatt-hours), tolerance for less-developed markets if "
            "the energy economics are right, and a preference for dedicated rather than "
            "shared infrastructure."
        ),
        "key_requirements": [
            "Lowest achievable power cost — $/MWh is the primary underwriting variable",
            "High renewable capacity factor for co-located clean energy sourcing",
            "Fast interconnection to get to power quickly",
            "Sufficient water for cooling at high power density",
            "Does NOT require existing tech ecosystem — willing to be first mover",
        ],
        "differentiating_signal": (
            "An AI lab's key question is: what is the all-in cost of power delivered "
            "to the rack? Markets like MISO Iowa and SPP Kansas, which rank lower on "
            "the baseline framework due to weak demand signals, become significantly "
            "more attractive when the customer's primary driver is energy cost."
        ),
    },

    "enterprise": {
        "label": "Enterprise Compute Cluster (Fortune 500 / Financial / Healthcare)",
        "pillar_weights": {
            "demand":      0.45,
            "energy":      0.20,
            "feasibility": 0.25,
            "strategic":   0.10,
        },
        "description": (
            "Large enterprises deploying dedicated private AI and HPC infrastructure "
            "for internal workloads (financial modeling, drug discovery, internal LLM "
            "deployment). Characterized by moderate scale (5–50 MW), latency sensitivity "
            "because workloads need to integrate with existing enterprise IT, preference "
            "for regulated utility markets with high reliability, and ESG reporting "
            "requirements that create demand for clean energy but not at the same "
            "intensity as a hyperscaler."
        ),
        "key_requirements": [
            "Geographic proximity to enterprise headquarters or data-generating operations",
            "High grid reliability — downtime has direct business impact",
            "Connectivity to existing enterprise IT networks and cloud on-ramps",
            "Regulatory stability — enterprises prefer markets with known permitting paths",
            "Clean energy preferred but procured via utility green tariffs, not complex PPAs",
        ],
        "differentiating_signal": (
            "Enterprise compute clusters follow enterprise headquarters, not energy resources. "
            "Northern Virginia, Chicago (PJM), and Dallas (ERCOT) score highly for enterprise "
            "customers despite weaker energy economics because the enterprise customer base "
            "is concentrated there. The tech employment index and fiber connectivity scores "
            "are the strongest predictors of enterprise demand."
        ),
    },
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_for_archetype(
    metric_scores: pd.DataFrame,
    archetype_id: str,
) -> pd.DataFrame:
    """
    Re-run pillar and composite scoring using an archetype's pillar weights
    instead of the baseline weights.

    Returns a DataFrame with columns: archetype_opportunity_score, archetype_rank,
    and the four pillar scores. Indexed by market_id.

    The metric-level normalization is unchanged — only the pillar aggregation
    weights differ. This isolates the effect of the archetype's priorities
    from the underlying market conditions.
    """
    if archetype_id not in ARCHETYPES:
        raise ValueError(
            f"Unknown archetype '{archetype_id}'. "
            f"Available: {list(ARCHETYPES.keys())}"
        )

    archetype_weights = ARCHETYPES[archetype_id]["pillar_weights"]

    # Temporarily substitute archetype weights into the config module so the
    # existing scoring functions pick them up without modification.
    import market_intelligence.src.config as cfg
    original = cfg.PILLAR_WEIGHTS.copy()
    cfg.PILLAR_WEIGHTS = archetype_weights

    try:
        pillar_scores = compute_pillar_scores(metric_scores)
        composite = compute_composite_score(pillar_scores)
        ranks = rank_markets(composite)
    finally:
        # Always restore baseline weights, even if scoring raises an exception.
        cfg.PILLAR_WEIGHTS = original

    result = pillar_scores.copy()
    result["opportunity_score"] = composite
    result["rank"] = ranks
    return result


def best_market_per_archetype(metric_scores: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer archetype, identify the top-ranked market.

    Returns a summary DataFrame with one row per archetype showing:
    archetype, top market, opportunity score, and the one-line rationale
    from the archetype's differentiating_signal.

    This is the "best market for each customer type" output.
    """
    rows = []
    for archetype_id, archetype_cfg in ARCHETYPES.items():
        scored = score_for_archetype(metric_scores, archetype_id)
        top = scored[scored["rank"] == 1]
        top_market_id = top.index[0]
        top_score = float(top.loc[top_market_id, "opportunity_score"])

        rows.append({
            "archetype":        archetype_cfg["label"],
            "top_market_id":    top_market_id,
            "opportunity_score": round(top_score, 1),
            "signal":           archetype_cfg["differentiating_signal"],
        })

    return pd.DataFrame(rows)


def archetype_ranking_comparison(
    metric_scores: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of market ranks under each archetype.

    Columns: market_name, baseline_rank, hyperscaler_rank, ai_lab_rank,
             enterprise_rank.

    Markets that move significantly between archetypes are the most
    strategically interesting — they are strongly aligned with one customer
    type but not others.
    """
    import market_intelligence.src.config as cfg

    # Baseline ranking
    original = cfg.PILLAR_WEIGHTS.copy()
    from market_intelligence.src.scoring import score_markets
    baseline = score_markets(raw_df, metric_scores)[["market_id", "market_name", "rank"]]
    baseline = baseline.rename(columns={"rank": "baseline_rank"})

    result = baseline.copy()

    for archetype_id in ARCHETYPES:
        scored = score_for_archetype(metric_scores, archetype_id)
        scored = scored[["rank"]].rename(columns={"rank": f"{archetype_id}_rank"})
        scored = scored.reset_index()  # bring market_id back as column
        result = result.merge(scored, on="market_id", how="left")

    return result.sort_values("baseline_rank").reset_index(drop=True)
