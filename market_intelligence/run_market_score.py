# market_intelligence/run_market_score.py
#
# CLI entry point for the market opportunity scoring pipeline.
#
# Usage (from project root):
#     python market_intelligence/run_market_score.py
#
# What it does:
#   1. Loads market_inputs.csv
#   2. Normalizes all metrics to 0–100 scores
#   3. Computes pillar scores and composite opportunity scores
#   4. Ranks markets from highest to lowest opportunity
#   5. Writes the scored table to data/processed/market_scores.csv
#   6. Prints a summary to the terminal

from __future__ import annotations

from pathlib import Path
import sys

# Allow running from the project root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from market_intelligence.src.processing import load_markets, normalize_all
from market_intelligence.src.scoring import score_markets
from market_intelligence.src.config import PILLAR_WEIGHTS, METRICS

# ── Paths ─────────────────────────────────────────────────────────────────────

MODULE_DIR   = Path(__file__).resolve().parent
INPUT_PATH   = MODULE_DIR / "data" / "raw"  / "market_inputs.csv"
OUTPUT_PATH  = MODULE_DIR / "data" / "processed" / "market_scores.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_summary(scored: pd.DataFrame) -> None:
    """Print a readable ranked summary to the terminal."""
    pillar_cols = [f"{p}_score" for p in PILLAR_WEIGHTS.keys()]
    display_cols = ["rank", "market_id", "market_name", "opportunity_score"] + pillar_cols

    # Only show columns that exist in the output
    display_cols = [c for c in display_cols if c in scored.columns]

    separator = "─" * 100
    print(f"\n{separator}")
    print("  AISRI — Market Opportunity Rankings")
    print(f"  {len(scored)} markets scored across "
          f"{len(METRICS)} metrics / {len(PILLAR_WEIGHTS)} pillars")
    print(separator)

    pd.set_option("display.float_format", "{:.1f}".format)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 120)
    print(scored[display_cols].to_string(index=False))
    print(separator)

    winner = scored.iloc[0]
    print(f"\n  Top market: {winner['market_name']} "
          f"(opportunity_score = {winner['opportunity_score']:.1f})\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[INFO] Loading:  {INPUT_PATH}")
    raw_df = load_markets(INPUT_PATH)
    print(f"[INFO] Loaded {len(raw_df)} markets, {len(METRICS)} metrics each.")

    # Step 1: Normalize all metrics to 0–100 scores
    print("[INFO] Normalizing metrics...")
    metric_scores = normalize_all(raw_df)

    # Step 2: Score markets (pillar aggregation → composite → rank)
    print("[INFO] Computing pillar and composite scores...")
    scored = score_markets(raw_df, metric_scores)

    # Step 3: Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK]   Wrote: {OUTPUT_PATH}")

    # Step 4: Print summary
    _print_summary(scored)


if __name__ == "__main__":
    main()
