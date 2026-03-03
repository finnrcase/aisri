# src/run_score_v1.py

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Project root setup
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import compute_scores  # noqa: E402
from src.confidence import ConfidenceConfig, compute_confidence  # noqa: E402


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def clamp100(x):
    if np.isnan(x):
        return np.nan
    return max(0.0, min(100.0, float(x)))


def weighted_mean(values, weights):
    vals = np.array([safe_float(v) for v in values], dtype=float)
    wts = np.array([safe_float(w) for w in weights], dtype=float)

    mask = ~np.isnan(vals) & (wts > 0)
    if mask.sum() == 0:
        return np.nan

    return float((vals[mask] * wts[mask]).sum() / wts[mask].sum())


# ------------------------------------------------------------------
# Scoring configuration (v2 deterministic aggregation)
# ------------------------------------------------------------------

PILLAR_COLS = [
    "pillar_energy",
    "pillar_efficiency",
    "pillar_carbon",
    "pillar_offsets",
    "pillar_transparency",
]

PILLAR_WEIGHTS = {
    "pillar_energy": 0.25,
    "pillar_efficiency": 0.20,
    "pillar_carbon": 0.25,
    "pillar_offsets": 0.15,
    "pillar_transparency": 0.15,
}

MISSING_PENALTY = 65.0  # institutional moderate penalty


# v1 confidence metric weights
CONF_METRIC_WEIGHTS = {
    "renewable_share_pct": 0.25,
    "pue": 0.20,
    "scope2_intensity": 0.25,
    "offset_share_scope2": 0.10,
    "reports_scope2_market_and_location": 0.05,
    "reports_electricity_consumption": 0.05,
    "reports_data_center_metrics": 0.05,
    "third_party_assurance_level": 0.05,
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    # -----------------------------
    # Load data
    # -----------------------------

    companies = pd.read_csv(ROOT / "data" / "raw" / "extracted" / "companies.csv")
    metrics = pd.read_csv(ROOT / "data" / "processed" / "metrics_final.csv")
    metric_defs = pd.read_csv(ROOT / "data" / "raw" / "extracted" / "metric_definitions.csv")

    sources_path = ROOT / "data" / "raw" / "extracted" / "sources.csv"
    sources = pd.read_csv(sources_path) if sources_path.exists() else None

    # -----------------------------
    # Base scoring (pillar-level)
    # -----------------------------

    scores = compute_scores(
        companies_df=companies,
        metrics_raw_df=metrics,
        metric_defs_df=metric_defs,
        fiscal_year=2024,
    )

    if "company_id" not in scores.columns:
        raise ValueError("compute_scores must output 'company_id'.")

    # -----------------------------
    # Deterministic overall_risk
    # (prevents NaNs from propagating)
    # -----------------------------

    # Ensure all pillar columns exist
    for col in PILLAR_COLS:
        if col not in scores.columns:
            scores[col] = np.nan

    # Apply missing-data penalty at pillar level
    for col in PILLAR_COLS:
        scores[col] = scores[col].apply(
            lambda x: MISSING_PENALTY if np.isnan(safe_float(x)) else clamp100(x)
        )

    # Compute overall risk
    def compute_overall(row):
        values = [row[c] for c in PILLAR_COLS]
        weights = [PILLAR_WEIGHTS[c] for c in PILLAR_COLS]
        v = weighted_mean(values, weights)
        return clamp100(v)

    scores["overall_risk"] = scores.apply(compute_overall, axis=1)

    # -----------------------------
    # Confidence integration
    # -----------------------------

    cfg = ConfidenceConfig()

    conf_rows = []
    for cid in scores["company_id"].tolist():

        cm = metrics[metrics["company_id"] == cid].copy()

        conf = compute_confidence(
            company_metrics=cm,
            metric_weights=CONF_METRIC_WEIGHTS,
            cfg=cfg,
            sources=sources,
            current_year=2024,
        )

        conf["company_id"] = cid
        conf_rows.append(conf)

    conf_df = pd.DataFrame(conf_rows)
    scores = scores.merge(conf_df, on="company_id", how="left")

    # -----------------------------
    # Save
    # -----------------------------

    out_path = ROOT / "data" / "processed" / "scores_v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    print(scores.sort_values("overall_risk").head(15))


if __name__ == "__main__":
    main()