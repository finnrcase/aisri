# src/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from src.normalize import V1_BOUNDS, linear_risk, binary_risk, assurance_risk


@dataclass(frozen=True)
class ScoreResult:
    company_id: str
    fiscal_year: int
    overall_risk: float
    pillar_risk: Dict[str, float]
    metric_risk: Dict[str, float]
    coverage: float  # share of required metrics present (0–1)


def compute_scores(
    companies_df: pd.DataFrame,
    metrics_raw_df: pd.DataFrame,
    metric_defs_df: pd.DataFrame,
    fiscal_year: int,
    missing_risk_default: float = 65.0,
) -> pd.DataFrame:
    """
    Produces one row per company with overall risk score and pillar breakdown.
    """

    # v1 weights (we can later import from config)
    pillar_weights = {
        "energy": 0.25,
        "efficiency": 0.20,
        "carbon": 0.25,
        "offsets": 0.10,
        "transparency": 0.20,
    }

    metric_weights = {
        "energy": {"renewable_share_pct": 1.0},
        "efficiency": {"pue": 1.0},
        "carbon": {"scope2_intensity": 1.0},
        "offsets": {"offset_share_scope2": 1.0},
        "transparency": {
            "reports_scope2_market_and_location": 0.25,
            "reports_electricity_consumption": 0.25,
            "reports_data_center_metrics": 0.25,
            "third_party_assurance_level": 0.25,
        },
    }

    required_metrics = [
        "renewable_share_pct",
        "pue",
        "scope2_intensity",
        "offset_share_scope2",
        "reports_scope2_market_and_location",
        "reports_electricity_consumption",
        "reports_data_center_metrics",
        "third_party_assurance_level",
    ]

    defs = metric_defs_df.set_index("metric_id").to_dict(orient="index")

    # Filter to year
    year_df = metrics_raw_df[metrics_raw_df["fiscal_year"] == fiscal_year].copy()

    out_rows = []
    for company_id in companies_df["company_id"].tolist():
        cdf = year_df[year_df["company_id"] == company_id]

        metric_risk: Dict[str, float] = {}
        present = set(cdf["metric_id"].tolist())

        # Coverage: only required metrics
        coverage = len(present.intersection(required_metrics)) / float(len(required_metrics))

        # Compute risk per required metric (fallback to missing policy)
        for m in required_metrics:
            if m not in present:
                # Transparency missing => treat as not disclosed
                if defs.get(m, {}).get("pillar") == "transparency":
                    metric_risk[m] = 100.0
                else:
                    metric_risk[m] = float(missing_risk_default)
                continue

            # Take first observed row (v1). Later we will apply hierarchy rules.
            row = cdf[cdf["metric_id"] == m].iloc[0]
            raw_val = row["value"]

            if m in ("reports_scope2_market_and_location", "reports_electricity_consumption", "reports_data_center_metrics"):
                # Handle missing values before casting
                if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)) or pd.isna(raw_val):
                    metric_risk[m] = 100.0  # missing transparency -> max risk
                else:
                    metric_risk[m] = binary_risk(int(float(raw_val)))
            elif m == "third_party_assurance_level":
                # Missing assurance -> treat as "none" (highest risk for assurance metric)
                if raw_val is None or pd.isna(raw_val):
                    metric_risk[m] = assurance_risk(0)
                else:
                    metric_risk[m] = assurance_risk(int(float(raw_val)))
            elif m in V1_BOUNDS:
                direction = defs[m]["directionality"]
                metric_risk[m] = linear_risk(float(str(raw_val)), V1_BOUNDS[m], direction)  # 0–100
            else:
                metric_risk[m] = float(missing_risk_default)

        # Pillar aggregation
        pillar_risk: Dict[str, float] = {}
        for pillar, w_map in metric_weights.items():
            total_w = sum(w_map.values())
            if total_w <= 0:
                pillar_risk[pillar] = float("nan")
                continue
            pillar_risk[pillar] = sum(metric_risk[m] * w for m, w in w_map.items()) / total_w

        overall = sum(pillar_risk[p] * pillar_weights[p] for p in pillar_weights.keys())

        out_rows.append(
            {
                "company_id": company_id,
                "fiscal_year": fiscal_year,
                "overall_risk": round(overall, 2),
                "coverage": round(coverage, 3),
                **{f"pillar_{k}": round(v, 2) for k, v in pillar_risk.items()},
            }
        )

    return pd.DataFrame(out_rows).sort_values("overall_risk", ascending=True).reset_index(drop=True)
