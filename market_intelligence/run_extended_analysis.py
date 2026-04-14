# market_intelligence/run_extended_analysis.py
#
# Extended analysis entry point.
# Runs the baseline scoring pipeline plus four additional layers:
#   1. Customer targeting  — best market per archetype
#   2. Scenario analysis   — ranking stability across structural scenarios
#   3. Risk assessment     — regulatory / grid / execution risk matrix
#   4. Project economics   — lightweight cost-revenue-payback summary
#
# Usage (from project root):
#   python market_intelligence/run_extended_analysis.py
#
# Outputs written to:
#   data/processed/market_scores.csv        (baseline, same as run_market_score.py)
#   outputs/tables/archetype_rankings.csv
#   outputs/tables/scenario_comparison.csv
#   outputs/tables/robust_markets.csv
#   outputs/tables/risk_matrix.csv
#   outputs/tables/economics_summary.csv
#   outputs/tables/opportunity_risk_quadrants.csv
#   outputs/figures/  (all existing figures + 3 new ones)

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from market_intelligence.src.processing  import load_markets, normalize_all
from market_intelligence.src.scoring     import score_markets
from market_intelligence.src.customer_targeting import (
    best_market_per_archetype,
    archetype_ranking_comparison,
)
from market_intelligence.src.scenarios   import (
    scenario_rank_comparison,
    most_robust_markets,
)
from market_intelligence.src.risk_assessment import (
    risk_matrix,
    flag_opportunity_risk_quadrants,
)
from market_intelligence.src.economics  import economics_summary
from market_intelligence.src.visualization import generate_all, export_full_results
from market_intelligence.src.visualization_extended import generate_extended_outputs

MODULE_DIR  = Path(__file__).resolve().parent
INPUT_PATH  = MODULE_DIR / "data" / "raw"  / "market_inputs.csv"
SCORES_PATH = MODULE_DIR / "data" / "processed" / "market_scores.csv"
TABLES_DIR  = MODULE_DIR / "outputs" / "tables"
FIGURES_DIR = MODULE_DIR / "outputs" / "figures"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Baseline scoring ──────────────────────────────────────────────
    print("\n[1/6] Running baseline scoring pipeline...")
    raw_df        = load_markets(INPUT_PATH)
    metric_scores = normalize_all(raw_df)
    scored_df     = score_markets(raw_df, metric_scores)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(SCORES_PATH, index=False)
    # Also write to outputs/tables/ so all tabular outputs are co-located
    scored_df.to_csv(TABLES_DIR / "market_scores.csv", index=False)
    print(f"      Baseline scores written to: {SCORES_PATH}")

    # ── Step 2: Customer targeting ────────────────────────────────────────────
    print("\n[2/6] Running customer targeting layer...")

    best = best_market_per_archetype(metric_scores)
    best.to_csv(TABLES_DIR / "archetype_best_markets.csv", index=False)
    print("      Best market per archetype:")
    for _, row in best.iterrows():
        print(f"        {row['archetype']:45s}  →  {row['top_market_id']}  ({row['opportunity_score']:.1f})")

    archetype_ranks = archetype_ranking_comparison(metric_scores, raw_df)
    archetype_ranks.to_csv(TABLES_DIR / "archetype_rankings.csv", index=False)

    # ── Step 3: Scenario analysis ─────────────────────────────────────────────
    print("\n[3/6] Running scenario analysis...")
    scenario_comparison = scenario_rank_comparison(raw_df)
    scenario_comparison.to_csv(TABLES_DIR / "scenario_comparison.csv", index=False)

    robust = most_robust_markets(raw_df)
    robust.to_csv(TABLES_DIR / "robust_markets.csv", index=False)
    print(f"      Most robust markets (stable across all scenarios):")
    for _, row in robust.iterrows():
        print(f"        {row['market_name']:30s}  avg rank: {row['avg_rank']:.1f}  worst rank: {int(row['worst_rank'])}")

    # ── Step 4: Risk assessment ───────────────────────────────────────────────
    print("\n[4/6] Running risk assessment...")
    risk_df = risk_matrix()
    risk_df.to_csv(TABLES_DIR / "risk_matrix.csv", index=False)

    quadrants = flag_opportunity_risk_quadrants(scored_df)
    quadrants_out = quadrants[
        ["market_id", "market_name", "opportunity_score", "rank",
         "aggregate_risk", "opportunity_risk_quadrant"]
    ].sort_values("rank")
    quadrants_out.to_csv(TABLES_DIR / "opportunity_risk_quadrants.csv", index=False)
    print("      Opportunity / Risk quadrant classification:")
    for _, row in quadrants_out.iterrows():
        print(f"        {row['market_name']:30s}  {row['opportunity_risk_quadrant']}")

    # ── Step 5: Project economics ─────────────────────────────────────────────
    print("\n[5/6] Computing project economics (100 MW reference project)...")
    econ = economics_summary(capacity_mw=100)
    econ.to_csv(TABLES_DIR / "economics_summary.csv", index=False)
    display_cols = ["market_id", "total_capex_mm", "annual_power_cost_mm",
                    "annual_revenue_mm", "annual_noi_mm", "simple_payback_years"]
    print(econ[display_cols].to_string(index=False))

    # ── Step 6: Figures and full results export ───────────────────────────────
    print("\n[6/6] Generating figures and full results export...")
    generate_all(scored_df, figures_dir=FIGURES_DIR, tables_dir=TABLES_DIR)
    generate_extended_outputs(
        scored_df=scored_df,
        raw_df=raw_df,
        metric_scores=metric_scores,
        figures_dir=FIGURES_DIR,
        tables_dir=TABLES_DIR,
    )

    # Comprehensive single-file export: all markets, all scored columns,
    # risk classification, quadrant, and recommendation label in one CSV.
    full_results_path = TABLES_DIR / "full_results.csv"
    full_results = export_full_results(scored_df, full_results_path)
    print(f"\n      Full results export: {full_results_path}")
    print(full_results[["rank", "market_name", "Opportunity Score",
                         "aggregate_risk", "recommendation"]].to_string(index=False))

    print("\n── All outputs written ──────────────────────────────────────────")
    print(f"   Figures:  {FIGURES_DIR}")
    print(f"   Tables:   {TABLES_DIR}")
    print(f"   Key file: {full_results_path}\n")


if __name__ == "__main__":
    main()
