# aisri/src/confidence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    Confidence is a DATA QUALITY score (coverage + source quality + recency + assurance),
    not a sustainability performance score.

    Default weights are conservative for v1:
      - coverage dominates
      - source quality matters a lot
      - recency matters somewhat
      - assurance is a small modifier (to avoid "laundering" weak disclosure)
    """
    w_coverage: float = 0.45
    w_source_quality: float = 0.30
    w_recency: float = 0.15
    w_assurance: float = 0.10

    # Penalty for imputation share (weighted). Conservative: up to 30 points off.
    max_imputation_penalty: float = 30.0

    # If no year column exists or is unusable, fall back to this recency score.
    default_recency_score: float = 50.0

    # Grade thresholds (inclusive lower bounds)
    grade_A: float = 80.0
    grade_B: float = 65.0
    grade_C: float = 45.0


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def confidence_grade(score_0_100: float, cfg: ConfidenceConfig) -> str:
    if score_0_100 >= cfg.grade_A:
        return "A"
    if score_0_100 >= cfg.grade_B:
        return "B"
    if score_0_100 >= cfg.grade_C:
        return "C"
    return "D"


def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _safe_float(x) -> Optional[float]:
    if _is_missing(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _year_col(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer fiscal_year (project standard). Fall back to year if present.
    """
    if "fiscal_year" in df.columns:
        return "fiscal_year"
    if "year" in df.columns:
        return "year"
    return None


def _pick_latest_row(rows: pd.DataFrame) -> pd.Series:
    """
    Choose the "latest" row for a metric if multiple rows exist:
      - prefer max fiscal_year/year when available and parseable
      - otherwise fall back to first row
    """
    yc = _year_col(rows)
    if yc is None or rows.empty:
        return rows.iloc[0]

    tmp = rows.copy()
    tmp["_year_num"] = pd.to_numeric(tmp[yc], errors="coerce")

    if tmp["_year_num"].notna().any():
        tmp = tmp.sort_values("_year_num", ascending=False)
        return tmp.iloc[0]

    return rows.iloc[0]


def compute_weighted_coverage(
    company_metrics: pd.DataFrame,
    metric_weights: Dict[str, float],
) -> Tuple[float, float]:
    """
    Returns:
      weighted_coverage in [0,1]
      imputed_weight_share in [0,1]

    Definition (v1):
      - A metric is "present" if a row exists AND the latest row for that metric has a non-null value.
      - Missing (no row or null value) -> treated as imputed for confidence purposes.
    """
    total_w = 0.0
    present_w = 0.0
    imputed_w = 0.0

    for metric_id, w in metric_weights.items():
        total_w += w
        rows = company_metrics[company_metrics["metric_id"] == metric_id]

        if rows.empty:
            imputed_w += w
            continue

        row = _pick_latest_row(rows)
        val = row.get("value", None)

        if _is_missing(val):
            imputed_w += w
        else:
            present_w += w

    weighted_coverage = 0.0 if total_w == 0 else (present_w / total_w)
    imputed_weight_share = 0.0 if total_w == 0 else (imputed_w / total_w)
    return weighted_coverage, imputed_weight_share


def compute_source_quality_score(
    company_metrics: pd.DataFrame,
    metric_weights: Dict[str, float],
    sources: Optional[pd.DataFrame] = None,
    source_quality_col: str = "quality_score",
    default_quality: float = 40.0,
) -> float:
    """
    Weighted mean source quality across PRESENT metrics.

    v1 behavior:
      - If sources is missing OR no (source_id -> sources) linkage OR no quality_score column:
        return default_quality (conservative, avoids false precision).
      - If available, compute weighted average quality score using the latest row per metric.
    """
    # If we can't link sources robustly, fall back to default
    if sources is None:
        return default_quality
    if "source_id" not in company_metrics.columns:
        return default_quality
    if "source_id" not in sources.columns:
        return default_quality
    if source_quality_col not in sources.columns:
        return default_quality

    src = sources.copy()
    src[source_quality_col] = pd.to_numeric(src[source_quality_col], errors="coerce")

    weighted_sum = 0.0
    total_w = 0.0

    for metric_id, w in metric_weights.items():
        rows = company_metrics[company_metrics["metric_id"] == metric_id]
        if rows.empty:
            continue

        row = _pick_latest_row(rows)
        val = row.get("value", None)
        if _is_missing(val):
            continue

        sid = row.get("source_id", None)
        q = default_quality

        if not _is_missing(sid):
            match = src[src["source_id"] == sid]
            if not match.empty:
                qv = match.iloc[0].get(source_quality_col, None)
                if not _is_missing(qv):
                    q = float(qv)

        weighted_sum += w * q
        total_w += w

    if total_w == 0:
        return default_quality

    return clamp(weighted_sum / total_w)


def compute_recency_score(
    company_metrics: pd.DataFrame,
    metric_weights: Dict[str, float],
    current_year: int,
    default_recency_score: float = 50.0,
) -> float:
    """
    Weighted mean recency across PRESENT metrics.

    Uses fiscal_year if available, otherwise year.

    Schedule:
      - same year: 100
      - 1 year old: 80
      - 2 years old: 60
      - 3+ years old: 40

    If no year column exists: default_recency_score.
    """
    yc = _year_col(company_metrics)
    if yc is None:
        return default_recency_score

    weighted_sum = 0.0
    total_w = 0.0

    for metric_id, w in metric_weights.items():
        rows = company_metrics[company_metrics["metric_id"] == metric_id]
        if rows.empty:
            continue

        row = _pick_latest_row(rows)
        val = row.get("value", None)
        if _is_missing(val):
            continue

        y = pd.to_numeric(row.get(yc, None), errors="coerce")
        if pd.isna(y):
            rec = default_recency_score
        else:
            age = max(0, current_year - int(y))
            if age == 0:
                rec = 100.0
            elif age == 1:
                rec = 80.0
            elif age == 2:
                rec = 60.0
            else:
                rec = 40.0

        weighted_sum += w * rec
        total_w += w

    if total_w == 0:
        return default_recency_score

    return clamp(weighted_sum / total_w)


def compute_assurance_component(
    company_metrics: pd.DataFrame,
    assurance_metric_id: str = "third_party_assurance_level",
) -> float:
    """
    Maps assurance level (0/1/2) to a confidence component.

    Mapping (conservative):
      0 -> 40
      1 -> 70
      2 -> 90
      missing -> 40
    """
    rows = company_metrics[company_metrics["metric_id"] == assurance_metric_id]
    if rows.empty:
        return 40.0

    row = _pick_latest_row(rows)
    v = _safe_float(row.get("value", None))

    if v is None:
        return 40.0
    if v >= 2:
        return 90.0
    if v >= 1:
        return 70.0
    return 40.0


def compute_confidence(
    company_metrics: pd.DataFrame,
    metric_weights: Dict[str, float],
    cfg: ConfidenceConfig,
    sources: Optional[pd.DataFrame] = None,
    current_year: int = 2024,
) -> Dict[str, Union[float, str]]:
    """
    Returns:
      confidence_score (0-100)
      confidence_grade (A-D)
      weighted_coverage (0-1)
      imputed_weight_share (0-1)
      coverage_score (0-100)
      source_quality_score (0-100)
      recency_score (0-100)
      assurance_score (0-100)
    """
    weighted_coverage, imputed_weight_share = compute_weighted_coverage(
        company_metrics=company_metrics,
        metric_weights=metric_weights,
    )

    # Coverage score penalizes imputation weight share.
    imputation_penalty = cfg.max_imputation_penalty * imputed_weight_share
    coverage_score = clamp(100.0 * weighted_coverage - imputation_penalty)

    source_quality_score = compute_source_quality_score(
        company_metrics=company_metrics,
        metric_weights=metric_weights,
        sources=sources,
    )

    recency_score = compute_recency_score(
        company_metrics=company_metrics,
        metric_weights=metric_weights,
        current_year=current_year,
        default_recency_score=cfg.default_recency_score,
    )

    assurance_score = compute_assurance_component(company_metrics=company_metrics)

    confidence_score = clamp(
        cfg.w_coverage * coverage_score
        + cfg.w_source_quality * source_quality_score
        + cfg.w_recency * recency_score
        + cfg.w_assurance * assurance_score
    )

    return {
        "confidence_score": float(confidence_score),
        "confidence_grade": confidence_grade(confidence_score, cfg),
        "weighted_coverage": float(weighted_coverage),
        "imputed_weight_share": float(imputed_weight_share),
        "coverage_score": float(coverage_score),
        "source_quality_score": float(source_quality_score),
        "recency_score": float(recency_score),
        "assurance_score": float(assurance_score),
    }
