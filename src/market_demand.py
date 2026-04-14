from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_TARGET_YEAR = 2030
NET_ZERO_PROXY_TARGET_SHARE_PCT = 90.0
SCENARIO_MULTIPLIERS: dict[str, float] = {
    "Base Case": 1.6,
    "High AI Demand Growth": 2.4,
    "Conservative Growth": 1.25,
}


def parse_energy_to_mwh(text: str) -> Optional[float]:
    if not text:
        return None

    cleaned = str(text).replace(",", "")

    million_kwh_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:million|mkwh)\s*k(?:ilo)?watt hours",
        cleaned,
        flags=re.IGNORECASE,
    )
    if million_kwh_match:
        return float(million_kwh_match.group(1)) * 1_000.0

    for unit, multiplier in [("TWh", 1_000_000.0), ("GWh", 1_000.0), ("MWh", 1.0)]:
        match = re.search(rf"([0-9]+(?:\.[0-9]+)?)\s*{unit}", cleaned, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) * multiplier

    plain_total_match = re.search(
        r"total electricity consumption[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if plain_total_match:
        return float(plain_total_match.group(1))

    return None


def extract_current_total_electricity_mwh(company_metrics: pd.DataFrame) -> tuple[Optional[float], str]:
    electricity_rows = company_metrics[company_metrics["metric_id"].astype(str) == "reports_electricity_consumption"].copy()
    if electricity_rows.empty:
        return None, "not_disclosed"

    text_fields: list[str] = []
    for _, row in electricity_rows.iterrows():
        for field in ["extraction_note", "method_note"]:
            value = row.get(field)
            if not pd.isna(value):
                text_fields.append(str(value))

    for text in text_fields:
        parsed = parse_energy_to_mwh(text)
        if parsed is not None:
            return parsed, "disclosed"

    for text in text_fields:
        inferred_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*GWh.*?equaling\s*([0-9]+(?:\.[0-9]+)?)%\s+of.*electricity use",
            text,
            flags=re.IGNORECASE,
        )
        if inferred_match:
            renewable_gwh = float(inferred_match.group(1))
            renewable_pct = float(inferred_match.group(2))
            if renewable_pct > 0:
                return renewable_gwh / (renewable_pct / 100.0) * 1_000.0, "inferred_from_note"

    return None, "not_disclosed"


def extract_current_renewable_share_pct(company_metrics: pd.DataFrame) -> tuple[Optional[float], str]:
    renewable_rows = company_metrics[company_metrics["metric_id"].astype(str) == "renewable_share_pct"].copy()
    if renewable_rows.empty:
        return None, "not_disclosed"

    value_series = renewable_rows["value"].dropna()
    if value_series.empty:
        return None, "not_disclosed"

    return float(value_series.iloc[0]), "disclosed"


def resolve_target_clean_energy_share_pct(
    company_metrics: pd.DataFrame,
    current_share_pct: Optional[float],
) -> tuple[Optional[float], Optional[int], str]:
    text_blob = " ".join(
        str(value)
        for column in ["extraction_note", "method_note"]
        if column in company_metrics.columns
        for value in company_metrics[column].dropna().astype(str).tolist()
    ).lower()

    year_match = re.search(r"\b(20[3-4][0-9])\b", text_blob)
    target_year = int(year_match.group(1)) if year_match else DEFAULT_TARGET_YEAR

    if "100% annual renewable energy matching" in text_blob or "100% of electricity consumed was matched with renewable energy" in text_blob:
        return 100.0, target_year, "explicit_100_percent_matching"

    explicit_target_match = re.search(r"(100|95|90)\s*%\s+(?:clean|renewable|carbon-free)", text_blob)
    if explicit_target_match:
        return float(explicit_target_match.group(1)), target_year, "explicit_clean_energy_target"

    if "net zero" in text_blob:
        return NET_ZERO_PROXY_TARGET_SHARE_PCT, DEFAULT_TARGET_YEAR, "net_zero_proxy_assumption"

    if current_share_pct is not None and current_share_pct >= 99.0:
        return 100.0, DEFAULT_TARGET_YEAR, "current_position_proxy"

    if current_share_pct is not None:
        return max(float(current_share_pct), 90.0), DEFAULT_TARGET_YEAR, "default_target_assumption"

    return 90.0, DEFAULT_TARGET_YEAR, "default_target_assumption"


def build_interpretation_tag(
    current_total_mwh: Optional[float],
    current_share_pct: Optional[float],
    clean_energy_gap_mwh: Optional[float],
    demand_basis: str,
) -> str:
    if current_total_mwh is None:
        return "High uncertainty due to limited electricity-use disclosure"

    if current_share_pct is not None and current_share_pct >= 95.0 and (clean_energy_gap_mwh is None or clean_energy_gap_mwh <= current_total_mwh * 0.1):
        return "Already relatively advanced on annual clean energy matching"

    if clean_energy_gap_mwh is not None and clean_energy_gap_mwh > current_total_mwh * 0.4:
        return "Large implied clean energy gap under growth assumptions"

    if clean_energy_gap_mwh is not None and clean_energy_gap_mwh > current_total_mwh * 0.15:
        return "Moderate clean energy build requirement under modeled demand growth"

    if demand_basis == "inferred_from_note":
        return "Model uses partial disclosure and should be treated as directional"

    return "Modeled clean energy requirement appears manageable relative to current position"


def classify_opportunity(
    current_total_mwh: Optional[float],
    clean_energy_gap_mwh: Optional[float],
    demand_basis: str,
    confidence_score: Optional[float],
    coverage_score: Optional[float],
) -> str:
    if current_total_mwh is None or demand_basis == "not_disclosed" or (coverage_score is not None and coverage_score < 50):
        return "Watchlist / incomplete disclosure"

    if clean_energy_gap_mwh is not None and current_total_mwh > 0:
        gap_ratio = clean_energy_gap_mwh / current_total_mwh
        if clean_energy_gap_mwh >= 2_000_000 or gap_ratio >= 0.35:
            return "Large clean energy procurement need"
        if current_total_mwh >= 5_000_000 and confidence_score is not None and confidence_score >= 60:
            return "Potential high-demand infrastructure customer"

    return "Lower near-term signal"


def build_company_market_narrative(
    company_name: str,
    current_total_mwh: Optional[float],
    current_share_pct: Optional[float],
    projected_total_mwh: Optional[float],
    clean_energy_gap_mwh: Optional[float],
    demand_basis: str,
    confidence_score: Optional[float],
    coverage_score: Optional[float],
    opportunity_classification: str,
) -> str:
    if current_total_mwh is None:
        return (
            f"{company_name} shows a potentially relevant infrastructure footprint, but current electricity use is not clearly disclosed. "
            f"That limits the confidence of any demand estimate and keeps the company in a watchlist posture rather than an actionable market signal."
        )

    current_twh = current_total_mwh / 1_000_000.0
    projected_twh = projected_total_mwh / 1_000_000.0 if projected_total_mwh is not None else None
    gap_twh = clean_energy_gap_mwh / 1_000_000.0 if clean_energy_gap_mwh is not None else None
    share_text = f"{current_share_pct:.0f}% clean energy share" if current_share_pct is not None else "an undisclosed clean energy share"
    confidence_text = (
        f"Confidence is supported by a score of {confidence_score:.0f} and coverage of {coverage_score:.0f}."
        if confidence_score is not None and coverage_score is not None
        else "Disclosure support is limited, so the estimate should be treated as directional."
    )

    if projected_twh is not None and gap_twh is not None:
        return (
            f"{company_name} currently appears to use about {current_twh:.2f} TWh of electricity with {share_text}. "
            f"Under the selected scenario, modeled demand rises to roughly {projected_twh:.2f} TWh, implying about {gap_twh:.2f} TWh of additional clean energy need versus today's disclosed position. "
            f"{confidence_text} From a business development perspective, this reads as a comparative infrastructure signal rather than a forecast, with the strongest relevance where large modeled demand coincides with credible disclosure and a meaningful clean energy gap. "
            f"Current classification: {opportunity_classification}."
        )

    return (
        f"{company_name} has a disclosed electricity base of about {current_twh:.2f} TWh with {share_text}, but the forward clean energy requirement remains uncertain. "
        f"{confidence_text} The company is best interpreted as a directional market signal rather than a clearly actionable infrastructure opportunity."
    )


def build_market_demand_table(
    scores_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    growth_multiplier = SCENARIO_MULTIPLIERS[scenario_name]
    records: list[dict] = []

    for _, score_row in scores_df.iterrows():
        company_id = str(score_row["company_id"])
        company_metrics = metrics_df[
            (metrics_df["company_id"].astype(str) == company_id)
            & (metrics_df["fiscal_year"] == score_row["fiscal_year"])
        ].copy() if "fiscal_year" in metrics_df.columns else metrics_df[metrics_df["company_id"].astype(str) == company_id].copy()

        current_total_mwh, demand_basis = extract_current_total_electricity_mwh(company_metrics)
        current_share_pct, current_share_basis = extract_current_renewable_share_pct(company_metrics)
        target_share_pct, target_year, target_basis = resolve_target_clean_energy_share_pct(company_metrics, current_share_pct)

        current_clean_mwh = (
            current_total_mwh * (current_share_pct / 100.0)
            if current_total_mwh is not None and current_share_pct is not None
            else np.nan
        )
        projected_total_mwh = (
            current_total_mwh * growth_multiplier
            if current_total_mwh is not None
            else np.nan
        )
        projected_required_clean_mwh = (
            projected_total_mwh * (target_share_pct / 100.0)
            if not pd.isna(projected_total_mwh) and target_share_pct is not None
            else np.nan
        )
        clean_energy_gap_mwh = (
            projected_required_clean_mwh - current_clean_mwh
            if not pd.isna(projected_required_clean_mwh) and not pd.isna(current_clean_mwh)
            else np.nan
        )

        interpretation_tag = build_interpretation_tag(
            current_total_mwh,
            current_share_pct,
            None if pd.isna(clean_energy_gap_mwh) else float(clean_energy_gap_mwh),
            demand_basis,
        )
        confidence_score = None if pd.isna(score_row.get("confidence_score")) else float(score_row.get("confidence_score"))
        coverage_score = None if pd.isna(score_row.get("coverage_score", score_row.get("coverage"))) else float(score_row.get("coverage_score", score_row.get("coverage")))
        opportunity_classification = classify_opportunity(
            current_total_mwh,
            None if pd.isna(clean_energy_gap_mwh) else float(clean_energy_gap_mwh),
            demand_basis,
            confidence_score,
            coverage_score,
        )
        market_narrative = build_company_market_narrative(
            score_row.get("company_name", company_id),
            current_total_mwh,
            current_share_pct,
            None if pd.isna(projected_total_mwh) else float(projected_total_mwh),
            None if pd.isna(clean_energy_gap_mwh) else float(clean_energy_gap_mwh),
            demand_basis,
            confidence_score,
            coverage_score,
            opportunity_classification,
        )

        records.append(
            {
                "company_id": company_id,
                "company_name": score_row.get("company_name", company_id),
                "fiscal_year": score_row.get("fiscal_year"),
                "scenario": scenario_name,
                "current_total_electricity_mwh": current_total_mwh,
                "current_renewable_share_pct": current_share_pct,
                "current_clean_energy_mwh": current_clean_mwh,
                "target_clean_energy_share_pct": target_share_pct,
                "target_year": target_year,
                "projected_demand_growth_multiplier": growth_multiplier,
                "projected_total_electricity_mwh": projected_total_mwh,
                "projected_required_clean_energy_mwh": projected_required_clean_mwh,
                "clean_energy_gap_mwh": clean_energy_gap_mwh,
                "demand_basis": demand_basis,
                "renewable_share_basis": current_share_basis,
                "target_basis": target_basis,
                "interpretation_tag": interpretation_tag,
                "opportunity_classification": opportunity_classification,
                "market_narrative": market_narrative,
                "assumption_note": (
                    f"Scenario multiplier: {growth_multiplier:.2f}x. "
                    f"Demand basis: {demand_basis}. "
                    f"Target basis: {target_basis.replace('_', ' ')}."
                ),
            }
        )

    demand_df = pd.DataFrame(records)
    if not demand_df.empty:
        demand_df = demand_df.sort_values("clean_energy_gap_mwh", ascending=False, na_position="last").reset_index(drop=True)
    return demand_df
