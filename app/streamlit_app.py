# app/streamlit_app.py
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.market_demand import (
        DEFAULT_TARGET_YEAR,
        NET_ZERO_PROXY_TARGET_SHARE_PCT,
        SCENARIO_MULTIPLIERS,
        build_market_demand_table,
    )
except ModuleNotFoundError:
    from market_demand import (
        DEFAULT_TARGET_YEAR,
        NET_ZERO_PROXY_TARGET_SHARE_PCT,
        SCENARIO_MULTIPLIERS,
        build_market_demand_table,
    )
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "dashboard_exports"

SCORES_PATH = DATA_DIR / "processed" / "scores_v1.csv"
METRICS_PATH = DATA_DIR / "processed" / "metrics_final.csv"
COMPANIES_PATH = DATA_DIR / "raw" / "extracted" / "companies.csv"
SOURCES_PATH = DATA_DIR / "raw" / "extracted" / "sources.csv"
METRIC_DEFS_PATH = DATA_DIR / "raw" / "extracted" / "metric_definitions.csv"

PILLAR_LABELS = {
    "pillar_energy": "Energy",
    "pillar_efficiency": "Efficiency",
    "pillar_carbon": "Carbon",
    "pillar_offsets": "Offsets",
    "pillar_transparency": "Transparency",
}

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#8FA7C6"
ACCENT_ALT = "#5B708C"
GRID = "rgba(255,255,255,0.10)"
TEXT = "#E8EEF5"


st.set_page_config(
    page_title="AISRI | Sustainability Risk Dashboard",
    page_icon="AI",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }
      .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        background:
          radial-gradient(circle at top right, rgba(143,167,198,0.20), transparent 30%),
          linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        margin-bottom: 1rem;
      }
      .eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem; color: #8FA7C6; margin-bottom: 0.45rem; }
      .hero-title { font-size: 2.15rem; font-weight: 700; line-height: 1.05; margin: 0 0 0.5rem 0; color: #E8EEF5; }
      .hero-copy { color: #9BA9B8; max-width: 880px; font-size: 1rem; line-height: 1.6; margin: 0; }
      .section-title { font-size: 1.1rem; font-weight: 650; color: #E8EEF5; margin-top: 0.4rem; margin-bottom: 0.25rem; }
      .section-copy { color: #9BA9B8; margin-bottom: 0.9rem; }
      .card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        background: rgba(255,255,255,0.04);
        min-height: 118px;
      }
      .card-label { color: #9BA9B8; font-size: 0.84rem; margin-bottom: 0.45rem; }
      .card-value { color: #E8EEF5; font-size: 1.6rem; font-weight: 700; line-height: 1.05; margin-bottom: 0.35rem; }
      .card-note { color: #9BA9B8; font-size: 0.84rem; }
      .panel {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem 1rem;
        background: rgba(255,255,255,0.04);
      }
      .insight-list { margin: 0; padding-left: 1.15rem; color: #E8EEF5; }
      .insight-list li { margin-bottom: 0.55rem; line-height: 1.45; }
      .divider { height: 1px; background: rgba(255,255,255,0.08); margin: 1rem 0 1.25rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not _exists(SCORES_PATH) or not _exists(METRICS_PATH):
        return None, None, None, None, None

    scores = _load_csv(str(SCORES_PATH), _file_mtime(SCORES_PATH))
    metrics = _load_csv(str(METRICS_PATH), _file_mtime(METRICS_PATH))
    companies = _load_csv(str(COMPANIES_PATH), _file_mtime(COMPANIES_PATH)) if _exists(COMPANIES_PATH) else None
    sources = _load_csv(str(SOURCES_PATH), _file_mtime(SOURCES_PATH)) if _exists(SOURCES_PATH) else None
    metric_defs = _load_csv(str(METRIC_DEFS_PATH), _file_mtime(METRIC_DEFS_PATH)) if _exists(METRIC_DEFS_PATH) else None
    return scores, metrics, companies, sources, metric_defs


def fmt_num(value, nd: int = 2) -> str:
    try:
        if pd.isna(value):
            return "Not disclosed"
        return f"{float(value):,.{nd}f}"
    except Exception:
        return "Not disclosed"


def fmt_pct(value, nd: int = 0) -> str:
    try:
        if pd.isna(value):
            return "Not disclosed"
        return f"{float(value):.{nd}f}%"
    except Exception:
        return "Not disclosed"


def fmt_ratio(value, nd: int = 2) -> str:
    try:
        if pd.isna(value):
            return "Not disclosed"
        return f"{float(value):.{nd}f}"
    except Exception:
        return "Not disclosed"


def fmt_intensity_tco2_per_mwh(value, nd: int = 5) -> str:
    try:
        if pd.isna(value):
            return "Not disclosed"
        return f"{float(value):.{nd}f} tCO2/MWh"
    except Exception:
        return "Not disclosed"


def truncate_text(value, limit: int = 120) -> str:
    if pd.isna(value):
        return "Not disclosed"
    text = str(value).strip()
    if not text:
        return "Not disclosed"
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def make_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value">{value}</div>
          <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_section_header(title: str, copy: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


def readable_metric_name(metric_id: str) -> str:
    return str(metric_id).replace("_", " ").strip().title()


def format_metric_value(metric_id: str, value) -> str:
    metric = str(metric_id)
    if pd.isna(value):
        return "Not disclosed"
    if metric in {"renewable_share_pct", "offset_share_scope2"}:
        return fmt_pct(value, 0)
    if metric == "pue":
        return fmt_ratio(value, 2)
    if metric == "scope2_intensity":
        return fmt_intensity_tco2_per_mwh(value, 5)
    if metric.startswith("reports_"):
        return "Yes" if float(value) == 1.0 else "No"
    if metric == "third_party_assurance_level":
        return str(int(float(value)))
    return fmt_num(value, 4)


def prepare_underlying_metrics(metrics_df: pd.DataFrame, metric_defs_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    label_map: dict[str, str] = {}
    if metric_defs_df is not None and {"metric_id", "label"}.issubset(metric_defs_df.columns):
        label_map = dict(zip(metric_defs_df["metric_id"].astype(str), metric_defs_df["label"].astype(str)))

    table = metrics_df.copy()
    table["Metric"] = table["metric_id"].astype(str).map(label_map).fillna(table["metric_id"].astype(str).map(readable_metric_name))
    table["Reported Value"] = table.apply(lambda row: format_metric_value(row["metric_id"], row["value"]), axis=1)
    table["Unit"] = table["unit"].apply(lambda value: "Not disclosed" if pd.isna(value) or str(value).strip() == "" else str(value))
    table["Source"] = table["source_id"].apply(lambda value: "Not disclosed" if pd.isna(value) or str(value).strip() == "" else str(value))
    note_series = table["extraction_note"] if "extraction_note" in table.columns else pd.Series(index=table.index, dtype="object")
    table["Notes"] = note_series.apply(truncate_text)
    table["Method"] = table["method_note"].apply(lambda value: "Not disclosed" if pd.isna(value) or str(value).strip() == "" else truncate_text(value, 90)) if "method_note" in table.columns else "Not disclosed"
    return table[["Metric", "Reported Value", "Unit", "Source", "Method", "Notes"]].sort_values("Metric")


def build_key_insights(row: dict, metrics_df: pd.DataFrame) -> list[str]:
    insights: list[str] = []

    overall_risk = row.get("overall_risk")
    confidence_grade = str(row.get("confidence_grade", "")).strip().upper()
    confidence_score = row.get("confidence_score")
    coverage = row.get("coverage_score", row.get("coverage"))

    if not pd.isna(overall_risk):
        insights.append(f"Overall risk is {fmt_num(overall_risk, 2)}. Lower values indicate a comparatively stronger sustainability position.")

    if confidence_grade:
        insights.append(f"Confidence is graded {confidence_grade}, with a supporting confidence score of {fmt_num(confidence_score, 1)}.")

    if not pd.isna(coverage):
        insights.append(f"Coverage stands at {fmt_num(coverage, 1)}, reflecting how much company disclosure supports the assessment.")

    pillar_values = []
    for column, label in PILLAR_LABELS.items():
        value = row.get(column)
        if not pd.isna(value):
            pillar_values.append((label, float(value)))

    if pillar_values:
        best_label, best_value = min(pillar_values, key=lambda item: item[1])
        worst_label, worst_value = max(pillar_values, key=lambda item: item[1])
        insights.append(f"Relative strength is concentrated in {best_label} ({fmt_num(best_value, 2)}), while {worst_label} ({fmt_num(worst_value, 2)}) is the largest pressure point.")

    if not metrics_df.empty and "value" in metrics_df.columns:
        disclosed = int(metrics_df["value"].notna().sum())
        insights.append(f"{disclosed} of {len(metrics_df)} tracked metrics are disclosed for this company-year profile.")

    return insights[:4]


def build_interpretation(row: dict) -> str:
    pillar_values = []
    for column, label in PILLAR_LABELS.items():
        value = row.get(column)
        if not pd.isna(value):
            pillar_values.append((label, float(value)))

    if not pillar_values:
        return "Pillar-level interpretation is not available for this company-year selection."

    strongest = min(pillar_values, key=lambda item: item[1])
    weakest = max(pillar_values, key=lambda item: item[1])
    confidence_grade = str(row.get("confidence_grade", "Not disclosed")).strip().upper() or "Not disclosed"
    coverage = fmt_num(row.get("coverage_score", row.get("coverage")), 1)
    return (
        f"The current profile is led by {strongest[0]}, which is the clearest relative strength in the model. "
        f"The main area to interrogate is {weakest[0]}, which contributes the highest risk score and is the most actionable weakness. "
        f"Confidence is {confidence_grade} with coverage at {coverage}, so the result is best read as a structured comparative view of disclosed performance rather than a full operational audit."
    )


def build_pillar_frame(row: dict) -> pd.DataFrame:
    records = []
    for column, label in PILLAR_LABELS.items():
        value = row.get(column)
        if not pd.isna(value):
            records.append({"Pillar": label, "Score": float(value)})
    return pd.DataFrame(records)


def style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color=TEXT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, title=None)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title=None)
    return fig


def build_pillar_chart(pillars_df: pd.DataFrame) -> go.Figure:
    chart = pillars_df.sort_values("Score", ascending=True)
    fig = px.bar(chart, x="Score", y="Pillar", orientation="h", text="Score", color_discrete_sequence=[ACCENT])
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    fig.update_layout(height=340, title="Pillar Risk Profile")
    return style_plotly(fig)


def build_company_comparison_chart(scores_df: pd.DataFrame, selected_company: str) -> go.Figure:
    frame = scores_df.copy()
    frame["Highlight"] = np.where(frame["company_id"] == selected_company, "Selected", "Peer")
    frame = frame.nsmallest(10, "overall_risk")
    fig = px.bar(
        frame.sort_values("overall_risk", ascending=False),
        x="overall_risk",
        y="company_name",
        orientation="h",
        color="Highlight",
        color_discrete_map={"Selected": ACCENT, "Peer": ACCENT_ALT},
        text="overall_risk",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    fig.update_layout(height=380, title="Peer Comparison: Lowest Risk Scores")
    return style_plotly(fig)


def build_risk_confidence_scatter(scores_df: pd.DataFrame, selected_company: str) -> go.Figure:
    frame = scores_df.dropna(subset=["overall_risk", "confidence_score"]).copy()
    frame["Focus"] = np.where(frame["company_id"] == selected_company, "Selected company", "Peer set")
    fig = px.scatter(
        frame,
        x="overall_risk",
        y="confidence_score",
        size="coverage_score" if "coverage_score" in frame.columns else None,
        color="Focus",
        hover_name="company_name",
        hover_data={"company_id": True, "overall_risk": ":.2f", "confidence_score": ":.2f"},
        color_discrete_map={"Selected company": ACCENT, "Peer set": "#64748B"},
    )
    fig.update_layout(height=360, title="Risk vs. Confidence")
    return style_plotly(fig)


def build_history_chart(scores_df: pd.DataFrame, chosen: str) -> Optional[go.Figure]:
    if "fiscal_year" not in scores_df.columns or "overall_risk" not in scores_df.columns:
        return None
    hist = scores_df[scores_df["company_id"] == chosen].dropna(subset=["fiscal_year", "overall_risk"]).copy()
    if hist["fiscal_year"].nunique() < 2:
        return None
    hist = hist.sort_values("fiscal_year")
    fig = px.line(hist, x="fiscal_year", y="overall_risk", markers=True)
    fig.update_traces(line_color=ACCENT, marker_color=ACCENT, hovertemplate="FY%{x}: %{y:.2f}<extra></extra>")
    fig.update_layout(height=320, title="Historical Risk Trend")
    return style_plotly(fig)


def build_gap_ranking_chart(demand_df: pd.DataFrame) -> go.Figure:
    frame = demand_df.dropna(subset=["clean_energy_gap_mwh"]).copy()
    frame["clean_energy_gap_twh"] = frame["clean_energy_gap_mwh"] / 1_000_000.0
    frame = frame.nlargest(10, "clean_energy_gap_twh").sort_values("clean_energy_gap_twh", ascending=True)
    fig = px.bar(
        frame,
        x="clean_energy_gap_twh",
        y="company_name",
        orientation="h",
        text="clean_energy_gap_twh",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(texttemplate="%{text:.2f} TWh", textposition="outside", hovertemplate="%{y}: %{x:.2f} TWh<extra></extra>")
    fig.update_layout(height=420, title="Largest Modeled Clean Power Procurement Gaps")
    return style_plotly(fig)


def build_energy_mix_chart(demand_df: pd.DataFrame) -> go.Figure:
    frame = demand_df.dropna(subset=["current_total_electricity_mwh"]).copy()
    frame["current_total_electricity_twh"] = frame["current_total_electricity_mwh"] / 1_000_000.0
    frame["current_clean_energy_twh"] = frame["current_clean_energy_mwh"] / 1_000_000.0
    frame["projected_total_electricity_twh"] = frame["projected_total_electricity_mwh"] / 1_000_000.0
    frame["projected_required_clean_energy_twh"] = frame["projected_required_clean_energy_mwh"] / 1_000_000.0
    frame = frame[["company_name", "current_clean_energy_twh", "current_total_electricity_twh", "projected_required_clean_energy_twh", "projected_total_electricity_twh"]].copy()
    frame["Current non-clean"] = (frame["current_total_electricity_twh"] - frame["current_clean_energy_twh"]).clip(lower=0)
    frame["Projected non-clean remainder"] = (frame["projected_total_electricity_twh"] - frame["projected_required_clean_energy_twh"]).clip(lower=0)
    plot_records = []
    for _, row in frame.iterrows():
        plot_records.extend(
            [
                {"company_name": row["company_name"], "Series": "Current clean", "TWh": row["current_clean_energy_twh"]},
                {"company_name": row["company_name"], "Series": "Current non-clean", "TWh": row["Current non-clean"]},
                {"company_name": row["company_name"], "Series": "Projected clean need", "TWh": row["projected_required_clean_energy_twh"]},
                {"company_name": row["company_name"], "Series": "Projected remaining mix", "TWh": row["Projected non-clean remainder"]},
            ]
        )
    plot_frame = pd.DataFrame(plot_records)
    fig = px.bar(
        plot_frame,
        x="company_name",
        y="TWh",
        color="Series",
        barmode="stack",
        color_discrete_map={
            "Current clean": ACCENT,
            "Current non-clean": "#546273",
            "Projected clean need": "#A8C0D8",
            "Projected remaining mix": "#39485A",
        },
    )
    fig.update_layout(height=420, title="Current Position vs. Modeled Future Clean Power Need")
    fig.update_xaxes(tickangle=-35)
    return style_plotly(fig)


def build_growth_disclosure_scatter(demand_df: pd.DataFrame) -> go.Figure:
    frame = demand_df.dropna(subset=["projected_demand_growth_multiplier"]).copy()
    frame["clean_energy_gap_twh"] = frame["clean_energy_gap_mwh"] / 1_000_000.0
    frame = frame.dropna(subset=["clean_energy_gap_twh"])
    fig = px.scatter(
        frame,
        x="projected_demand_growth_multiplier",
        y="clean_energy_gap_twh",
        color="demand_basis",
        size="projected_total_electricity_mwh",
        hover_name="company_name",
        hover_data={"target_clean_energy_share_pct": True, "target_year": True},
        color_discrete_map={"disclosed": ACCENT, "inferred_from_note": "#B9C6D4", "not_disclosed": "#64748B"},
    )
    fig.update_layout(height=360, title="Demand Growth vs. Procurement Gap Signal")
    fig.update_xaxes(title="Demand growth multiplier")
    return style_plotly(fig)


def build_market_headlines(demand_df: pd.DataFrame) -> list[str]:
    headlines: list[str] = []
    if demand_df.empty:
        return ["No modeled market-demand signals are available for the current selection."]

    modeled = demand_df.dropna(subset=["clean_energy_gap_mwh"]).copy()
    if not modeled.empty:
        largest_gap = modeled.iloc[0]
        headlines.append(
            f"{largest_gap['company_name']} shows the largest modeled clean energy procurement gap in the current scenario."
        )

    weak_disclosure = demand_df[
        (demand_df["demand_basis"] == "not_disclosed")
        & (demand_df["projected_total_electricity_mwh"].isna())
    ]
    if not weak_disclosure.empty:
        names = ", ".join(weak_disclosure["company_name"].head(3).tolist())
        headlines.append(
            f"{names} stand out as potentially relevant demand cases where electricity-use disclosure is still too limited for a defensible estimate."
        )

    high_demand = demand_df[
        demand_df["opportunity_classification"].isin(
            ["Potential high-demand infrastructure customer", "Large clean energy procurement need"]
        )
    ]
    if not high_demand.empty:
        names = ", ".join(high_demand["company_name"].head(3).tolist())
        headlines.append(
            f"{names} currently read as the strongest infrastructure partnership signals on a comparative basis."
        )

    lower_priority = demand_df[demand_df["opportunity_classification"] == "Lower near-term signal"]
    if not lower_priority.empty:
        names = ", ".join(lower_priority["company_name"].head(3).tolist())
        headlines.append(
            f"{names} appear lower priority for near-term infrastructure outreach because the modeled gap is smaller or the signal is less differentiated."
        )

    return headlines[:4]


def rankings_export_frame(scores_df: pd.DataFrame) -> pd.DataFrame:
    frame = scores_df.copy()
    columns = [
        "company_name",
        "company_id",
        "overall_risk",
        "confidence_score",
        "confidence_grade",
        "coverage_score",
        "pillar_energy",
        "pillar_efficiency",
        "pillar_carbon",
        "pillar_offsets",
        "pillar_transparency",
    ]
    available = [column for column in columns if column in frame.columns]
    export = frame[available].copy()
    return export.rename(
        columns={
            "company_name": "Company",
            "company_id": "Company ID",
            "overall_risk": "Overall Risk",
            "confidence_score": "Confidence Score",
            "confidence_grade": "Confidence Grade",
            "coverage_score": "Coverage",
            "pillar_energy": "Energy",
            "pillar_efficiency": "Efficiency",
            "pillar_carbon": "Carbon",
            "pillar_offsets": "Offsets",
            "pillar_transparency": "Transparency",
        }
    )


def save_export(df: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(df.to_csv(index=False), encoding="utf-8")
    return path


def export_controls(df: pd.DataFrame, label: str, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col_download, col_save = st.columns([1, 1])
    with col_download:
        st.download_button(
            label=f"Download {label} CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )
    with col_save:
        if st.button(f"Save {label} to outputs", use_container_width=True, key=f"save-{filename}"):
            saved_path = save_export(df, filename)
            st.success(f"Saved to {saved_path}")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI Infrastructure Sustainability Analytics</div>
      <div class="hero-title">AISRI Dashboard</div>
      <p class="hero-copy">
        A decision-support dashboard for evaluating disclosed sustainability risk across AI infrastructure companies.
        Explore company positioning, confidence, pillar-level performance, and supporting evidence in a structured executive view.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

scores, metrics, companies, sources, metric_defs = load_data()

if scores is None or metrics is None:
    st.error("Missing required files. Run `python src/build_metrics_final.py` and `python src/run_score_v1.py`, then reload the app.")
    st.stop()

name_map: dict[str, str] = {}
if companies is not None and {"company_id", "company_name"}.issubset(companies.columns):
    name_map = dict(zip(companies["company_id"].astype(str), companies["company_name"].astype(str)))

scores = scores.copy()
scores["company_id"] = scores["company_id"].astype(str)
scores["company_name"] = scores["company_id"].map(name_map).fillna(scores["company_id"])

st.sidebar.markdown("### Dashboard Controls")
available_years = sorted(scores["fiscal_year"].dropna().unique().tolist()) if "fiscal_year" in scores.columns else [2024]
year = st.sidebar.selectbox("Fiscal year", available_years, index=len(available_years) - 1)
view = st.sidebar.radio("View", ["Executive Dashboard", "Company Profile", "Methodology", "Data Sources"], index=0)
scenario_name = st.sidebar.selectbox("Demand scenario", list(SCENARIO_MULTIPLIERS.keys()), index=0)
search = st.sidebar.text_input("Search company", value="").strip().lower()

scores_y = scores.copy()
if "fiscal_year" in scores_y.columns:
    scores_y = scores_y[scores_y["fiscal_year"] == year].copy()
if search:
    search_mask = (
        scores_y["company_id"].astype(str).str.lower().str.contains(search)
        | scores_y["company_name"].astype(str).str.lower().str.contains(search)
    )
    scores_y = scores_y[search_mask].copy()
if "overall_risk" in scores_y.columns:
    scores_y = scores_y.sort_values(["overall_risk", "confidence_score"], ascending=[True, False], na_position="last").reset_index(drop=True)
else:
    scores_y = scores_y.sort_values(["confidence_score"], ascending=[False], na_position="last").reset_index(drop=True)
scores_y["rank"] = np.arange(1, len(scores_y) + 1)
demand_model_y = build_market_demand_table(scores_y, metrics, scenario_name)


def render_executive_dashboard() -> None:
    make_section_header(
        "Summary and Market View",
        "Compare company positioning for the selected year, then review the metrics, drivers, and supporting evidence behind each profile.",
    )

    total_companies = len(scores_y)
    scored_companies = int(scores_y["overall_risk"].notna().sum()) if "overall_risk" in scores_y.columns else 0
    avg_risk = fmt_num(scores_y["overall_risk"].mean(), 2) if "overall_risk" in scores_y.columns and scored_companies > 0 else "Not disclosed"
    avg_confidence = fmt_num(scores_y["confidence_score"].mean(), 1) if "confidence_score" in scores_y.columns else "Not disclosed"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        make_metric_card("Companies in view", str(total_companies), "Filtered set for the selected year")
    with c2:
        make_metric_card("Scored companies", str(scored_companies), "Companies with an overall risk output")
    with c3:
        make_metric_card("Average risk", avg_risk, "Lower values are better")
    with c4:
        make_metric_card("Average confidence", avg_confidence, "Model confidence across the current view")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    chart_col, insight_col = st.columns([1.7, 1.1], gap="large")
    with chart_col:
        top10 = scores_y.dropna(subset=["overall_risk"]).nsmallest(10, "overall_risk").copy()
        fig = px.bar(
            top10.sort_values("overall_risk", ascending=False),
            x="overall_risk",
            y="company_name",
            orientation="h",
            text="overall_risk",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>")
        fig.update_layout(height=420, title="Top 10 Companies by Lowest Sustainability Risk")
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    with insight_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Headline Insights")
        insights = []
        if not top10.empty:
            leader = top10.iloc[0]
            insights.append(f"{leader['company_name']} leads the current year with an overall risk score of {fmt_num(leader['overall_risk'], 2)}.")
        if "confidence_grade" in scores_y.columns:
            high_conf = scores_y[scores_y["confidence_grade"].isin(["A", "B"])]
            insights.append(f"{len(high_conf)} companies currently carry A or B confidence grades.")
        if "coverage_score" in scores_y.columns:
            low_coverage = scores_y[scores_y["coverage_score"].fillna(0) < 50]
            insights.append(f"{len(low_coverage)} companies have coverage below 50, suggesting disclosure limits may be material.")
        insights.append("Use the company profile view to inspect pillar drivers, interpretation, and the exportable underlying evidence.")
        st.markdown('<ul class="insight-list">' + "".join(f"<li>{item}</li>" for item in insights) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    make_section_header(
        "Comparative Table",
        "The full rankings remain available, but they are positioned as supporting evidence rather than the main experience.",
    )
    ranking_table = rankings_export_frame(scores_y)
    st.dataframe(ranking_table, use_container_width=True, hide_index=True)
    export_controls(ranking_table, "rankings", f"aisri_rankings_{year}.csv")

    make_section_header(
        "Market Demand & Infrastructure Signals",
        "A market-intelligence layer that estimates where future clean power procurement and infrastructure relevance may be strongest based on disclosed electricity use, renewable position, and transparent scenario assumptions.",
    )
    if demand_model_y.empty:
        st.info("No modeled demand data is available for the current selection.")
    else:
        top_gap = demand_model_y.iloc[0]
        modeled = demand_model_y.dropna(subset=["projected_total_electricity_mwh"]).copy()
        total_projected_twh = modeled["projected_total_electricity_mwh"].sum() / 1_000_000.0 if not modeled.empty else 0.0
        total_required_clean_twh = modeled["projected_required_clean_energy_mwh"].sum() / 1_000_000.0 if not modeled.empty else 0.0
        total_gap_twh = modeled["clean_energy_gap_mwh"].sum() / 1_000_000.0 if not modeled.empty else 0.0
        high_signal_count = int(
            demand_model_y["opportunity_classification"].isin(
                ["Potential high-demand infrastructure customer", "Large clean energy procurement need"]
            ).sum()
        )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            make_metric_card("Projected electricity", f"{total_projected_twh:.2f} TWh", "Across companies with modeled demand")
        with s2:
            make_metric_card("Required clean energy", f"{total_required_clean_twh:.2f} TWh", f"Scenario: {scenario_name}")
        with s3:
            make_metric_card("Modeled clean energy gap", f"{total_gap_twh:.2f} TWh", "Incremental clean procurement need")
        with s4:
            make_metric_card("High-signal companies", str(high_signal_count), "Comparative infrastructure opportunity signals")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        left_col, right_col = st.columns([1.6, 1.0], gap="large")
        with left_col:
            st.plotly_chart(build_gap_ranking_chart(demand_model_y), use_container_width=True)
        with right_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Headline market takeaways")
            largest_gap_twh = top_gap["clean_energy_gap_mwh"] / 1_000_000.0 if not pd.isna(top_gap["clean_energy_gap_mwh"]) else np.nan
            headlines = build_market_headlines(demand_model_y)
            st.markdown('<ul class="insight-list">' + "".join(f"<li>{item}</li>" for item in headlines) + "</ul>", unsafe_allow_html=True)
            st.caption(
                f"Modeled planning view only. The selected scenario applies a {SCENARIO_MULTIPLIERS[scenario_name]:.2f}x demand multiplier through {DEFAULT_TARGET_YEAR}. "
                f"Largest current modeled gap: {largest_gap_twh:.2f} TWh."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        mix_tab, scatter_tab, assumptions_tab = st.tabs(["Energy Mix", "Growth vs Gap", "Assumptions"])
        with mix_tab:
            st.plotly_chart(build_energy_mix_chart(demand_model_y.head(8)), use_container_width=True)
        with scatter_tab:
            st.plotly_chart(build_growth_disclosure_scatter(demand_model_y), use_container_width=True)
        with assumptions_tab:
            assumption_table = demand_model_y[
                [
                    "company_name",
                    "current_total_electricity_mwh",
                    "current_renewable_share_pct",
                    "target_clean_energy_share_pct",
                    "target_year",
                    "projected_demand_growth_multiplier",
                    "demand_basis",
                    "target_basis",
                    "opportunity_classification",
                ]
            ].copy()
            assumption_table.columns = [
                "Company",
                "Current Electricity (MWh)",
                "Current Clean Share (%)",
                "Target Clean Share (%)",
                "Target Year",
                "Demand Growth Multiplier",
                "Demand Basis",
                "Target Basis",
                "Opportunity Signal",
            ]
            assumption_table["Demand Growth Multiplier"] = assumption_table["Demand Growth Multiplier"].map(lambda x: f"{x:.2f}x")
            st.dataframe(assumption_table, use_container_width=True, hide_index=True)
            export_controls(demand_model_y, "market demand model", f"aisri_market_demand_{year}.csv")

        with st.expander("Assumptions and methodology note", expanded=False):
            st.markdown(
                f"""
                - Scenario selected: **{scenario_name}** using a **{SCENARIO_MULTIPLIERS[scenario_name]:.2f}x** demand multiplier through **{DEFAULT_TARGET_YEAR}**.
                - Current clean energy is modeled as `current_total_electricity_mwh * current_renewable_share_pct`.
                - If electricity use is not clearly disclosed, the app leaves projected demand and clean energy gap fields blank rather than inventing a baseline.
                - If an explicit clean-energy target is not found, the model uses a transparent default target assumption.
                - If only net-zero language is found, the model uses a proxy clean-energy target of **{NET_ZERO_PROXY_TARGET_SHARE_PCT:.0f}%** by **{DEFAULT_TARGET_YEAR}**.
                """
            )

        with st.expander("Underlying modeled table", expanded=False):
            st.dataframe(demand_model_y, use_container_width=True, hide_index=True)


def render_company_profile() -> None:
    options = scores_y.sort_values("company_name")["company_id"].astype(str).unique().tolist()
    chosen = st.selectbox("Select company", options, format_func=lambda cid: f"{name_map.get(cid, cid)} ({cid})")

    company_row = scores_y[scores_y["company_id"] == chosen].head(1)
    if company_row.empty:
        st.warning("No score row found for this company-year selection.")
        return

    row = company_row.iloc[0].to_dict()
    company_name = name_map.get(chosen, chosen)
    company_metrics = metrics[(metrics["company_id"].astype(str) == chosen) & (metrics["fiscal_year"] == year)].copy() if "fiscal_year" in metrics.columns else metrics[metrics["company_id"].astype(str) == chosen].copy()
    company_metrics["company_id"] = company_metrics["company_id"].astype(str)
    pillar_df = build_pillar_frame(row)
    insights = build_key_insights(row, company_metrics)
    demand_row = demand_model_y[demand_model_y["company_id"] == chosen].head(1)

    make_section_header(
        f"{company_name} | Company Profile",
        "A decision-support view of overall positioning, pillar-level drivers, peer context, interpretation, and raw supporting data.",
    )

    rank_value = int(company_row["rank"].iloc[0]) if "rank" in company_row.columns else None
    strongest = pillar_df.sort_values("Score", ascending=True).head(1)
    weakest = pillar_df.sort_values("Score", ascending=False).head(1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        make_metric_card("Overall risk", fmt_num(row.get("overall_risk"), 2), "Lower values indicate lower sustainability risk")
    with c2:
        make_metric_card("Confidence grade", str(row.get("confidence_grade", "Not disclosed")), f"Confidence score: {fmt_num(row.get('confidence_score'), 1)}")
    with c3:
        make_metric_card("Rank", f"#{rank_value}" if rank_value is not None else "Not disclosed", f"Out of {len(scores_y)} companies in view")
    with c4:
        label = strongest["Pillar"].iloc[0] if not strongest.empty else "Not disclosed"
        note = f"Top risk: {weakest['Pillar'].iloc[0]}" if not weakest.empty else ""
        make_metric_card("Top strength", label, note)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    top_col, summary_col = st.columns([1.55, 1.0], gap="large")
    with top_col:
        make_section_header("Summary", f"Fiscal year {year} | Company ID: {chosen}")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Headline insights")
        st.markdown('<ul class="insight-list">' + "".join(f"<li>{item}</li>" for item in insights) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with summary_col:
        make_section_header("Recommendation / Interpretation")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write(build_interpretation(row))
        st.markdown("</div>", unsafe_allow_html=True)

    make_section_header("Visual Analysis", "Charts are prioritized over raw tables so the dashboard reads like an analytical product rather than a worksheet.")
    chart_tab, comparison_tab, scatter_tab = st.tabs(["Pillar Breakdown", "Peer Comparison", "Risk vs Confidence"])

    with chart_tab:
        if pillar_df.empty:
            st.info("No pillar breakdown is available for this company-year selection.")
        else:
            st.plotly_chart(build_pillar_chart(pillar_df), use_container_width=True)

    with comparison_tab:
        peer_fig = build_company_comparison_chart(scores_y.dropna(subset=["overall_risk"]).copy(), chosen)
        st.plotly_chart(peer_fig, use_container_width=True)

    with scatter_tab:
        scatter_fig = build_risk_confidence_scatter(scores_y.copy(), chosen)
        st.plotly_chart(scatter_fig, use_container_width=True)

    history_fig = build_history_chart(scores.copy(), chosen)
    if history_fig is not None:
        make_section_header("Trend", "Historical context is shown when more than one fiscal year is available.")
        st.plotly_chart(history_fig, use_container_width=True)

    make_section_header(
        "Market Demand & Infrastructure Signals",
        "This market-research layer highlights where modeled clean power demand, infrastructure relevance, and disclosure quality may create stronger or weaker partnership signals.",
    )
    if demand_row.empty:
        st.info("No modeled market-demand estimate is available for this company.")
    else:
        d = demand_row.iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            current_total_text = f"{d['current_total_electricity_mwh'] / 1_000_000.0:.2f} TWh" if not pd.isna(d["current_total_electricity_mwh"]) else "Unavailable"
            make_metric_card("Current electricity", current_total_text, d["demand_basis"].replace("_", " ").title())
        with d2:
            projected_total_text = f"{d['projected_total_electricity_mwh'] / 1_000_000.0:.2f} TWh" if not pd.isna(d["projected_total_electricity_mwh"]) else "Unavailable"
            make_metric_card("Projected demand", projected_total_text, f"Through {int(d['target_year'])}")
        with d3:
            current_clean_text = f"{d['current_clean_energy_mwh'] / 1_000_000.0:.2f} TWh" if not pd.isna(d["current_clean_energy_mwh"]) else "Unavailable"
            renewable_share_text = f"{d['current_renewable_share_pct']:.0f}% current share" if not pd.isna(d["current_renewable_share_pct"]) else "Renewable share unavailable"
            make_metric_card("Current clean energy", current_clean_text, renewable_share_text)
        with d4:
            gap_text = f"{d['clean_energy_gap_mwh'] / 1_000_000.0:.2f} TWh" if not pd.isna(d["clean_energy_gap_mwh"]) else "Unavailable"
            make_metric_card("Opportunity signal", d["opportunity_classification"], gap_text)

        demand_chart_col, demand_note_col = st.columns([1.5, 1.0], gap="large")
        with demand_chart_col:
            if pd.isna(d["current_total_electricity_mwh"]):
                st.info("Current electricity use is not clearly disclosed, so the projected energy mix is not modeled for this company.")
            else:
                mix_df = pd.DataFrame(
                    {
                        "Stage": ["Current clean", "Current non-clean", "Projected clean need", "Projected remaining mix"],
                        "TWh": [
                            d["current_clean_energy_mwh"] / 1_000_000.0 if not pd.isna(d["current_clean_energy_mwh"]) else 0.0,
                            max((d["current_total_electricity_mwh"] - d["current_clean_energy_mwh"]) / 1_000_000.0, 0.0) if not pd.isna(d["current_clean_energy_mwh"]) else 0.0,
                            d["projected_required_clean_energy_mwh"] / 1_000_000.0 if not pd.isna(d["projected_required_clean_energy_mwh"]) else 0.0,
                            max((d["projected_total_electricity_mwh"] - d["projected_required_clean_energy_mwh"]) / 1_000_000.0, 0.0) if not pd.isna(d["projected_required_clean_energy_mwh"]) else 0.0,
                        ],
                    }
                )
                fig = px.bar(
                    mix_df,
                    x="Stage",
                    y="TWh",
                    text="TWh",
                    color="Stage",
                    color_discrete_map={
                        "Current clean": ACCENT,
                        "Current non-clean": "#546273",
                        "Projected clean need": "#A8C0D8",
                        "Projected remaining mix": "#39485A",
                    },
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig.update_layout(height=360, title="Current vs. Projected Energy Mix")
                st.plotly_chart(style_plotly(fig), use_container_width=True)
        with demand_note_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Business development interpretation")
            st.write(d["market_narrative"])
            st.write(
                f"Modeled target: {d['target_clean_energy_share_pct']:.0f}% clean energy by {int(d['target_year'])}. "
                f"Scenario multiplier: {d['projected_demand_growth_multiplier']:.2f}x."
            )
            st.caption(d["assumption_note"])
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Demand model assumptions for this company", expanded=False):
            st.markdown(
                f"""
                - Scenario: **{d['scenario']}**
                - Demand basis: **{d['demand_basis'].replace('_', ' ')}**
                - Target basis: **{str(d['target_basis']).replace('_', ' ')}**
                - If current electricity use is unavailable, the model does not fabricate a forecast.
                - Net-zero-only disclosures are proxied to **{NET_ZERO_PROXY_TARGET_SHARE_PCT:.0f}%** clean energy by **{DEFAULT_TARGET_YEAR}**.
                """
            )

    make_section_header("Export", "Download or save the curated outputs used in this view.")
    profile_export = rankings_export_frame(company_row)
    export_controls(profile_export, "company profile", f"aisri_{chosen}_{year}.csv")

    make_section_header("Supporting Data", "Detailed data remains available for transparency, but it is intentionally secondary to the narrative and charts above.")
    raw_tab, source_tab = st.tabs(["Underlying Metrics", "Source Coverage"])

    with raw_tab:
        if company_metrics.empty:
            st.info("No underlying metrics were found for this company-year selection.")
        else:
            clean_metrics = prepare_underlying_metrics(company_metrics, metric_defs)
            with st.expander("Show underlying metrics", expanded=False):
                st.dataframe(clean_metrics, use_container_width=True, hide_index=True)

    with source_tab:
        if company_metrics.empty:
            st.info("No source references are available because no underlying metrics were found.")
        else:
            source_view = company_metrics[["metric_id", "source_id", "quality_flag", "extraction_note"]].copy()
            source_view.columns = ["Metric", "Source", "Quality Flag", "Notes"]
            source_view["Metric"] = source_view["Metric"].map(readable_metric_name)
            source_view["Source"] = source_view["Source"].apply(lambda value: "Unavailable" if pd.isna(value) or str(value).strip() == "" else str(value))
            source_view["Quality Flag"] = source_view["Quality Flag"].apply(lambda value: "Unavailable" if pd.isna(value) or str(value).strip() == "" else str(value))
            source_view["Notes"] = source_view["Notes"].apply(truncate_text)
            with st.expander("Show source coverage", expanded=False):
                st.dataframe(source_view.sort_values("Metric"), use_container_width=True, hide_index=True)


def render_methodology() -> None:
    make_section_header(
        "Methodology",
        "AISRI is a comparative sustainability risk index for AI infrastructure companies. It is designed for structured comparison, not as a substitute for a full ESG audit.",
    )
    st.markdown(
        """
        - Energy sourcing evaluates renewable and carbon-free electricity exposure.
        - Efficiency captures data-center operational effectiveness.
        - Carbon captures scope 2 intensity and related exposure.
        - Offsets reflects reliance on offsets or contractual instruments where disclosed.
        - Transparency captures disclosure quality and assurance.
        """
    )

    with st.expander("Current output files", expanded=False):
        st.write(
            {
                "scores_v1.csv": str(SCORES_PATH),
                "metrics_final.csv": str(METRICS_PATH),
                "companies.csv": str(COMPANIES_PATH) if _exists(COMPANIES_PATH) else "(missing)",
                "metric_definitions.csv": str(METRIC_DEFS_PATH) if _exists(METRIC_DEFS_PATH) else "(missing)",
            }
        )

    with st.expander("Preview current-year scoring inputs", expanded=False):
        st.dataframe(rankings_export_frame(scores_y.head(25)), use_container_width=True, hide_index=True)


def render_sources() -> None:
    make_section_header(
        "Data Sources",
        "Source tables remain accessible for traceability, but are organized as reference material rather than the primary interface.",
    )
    if sources is None or sources.empty:
        st.info("No sources.csv file was found, or it is empty.")
        return

    source_table = sources.copy()
    keep = [column for column in ["source_id", "title", "url", "publisher", "year", "notes"] if column in source_table.columns]
    if keep:
        source_table = source_table[keep]
    source_table = source_table.rename(
        columns={
            "source_id": "Source ID",
            "title": "Title",
            "url": "URL",
            "publisher": "Publisher",
            "year": "Year",
            "notes": "Notes",
        }
    )
    if "Notes" in source_table.columns:
        source_table["Notes"] = source_table["Notes"].apply(truncate_text)
    st.dataframe(source_table, use_container_width=True, hide_index=True)
    export_controls(source_table, "sources", f"aisri_sources_{year}.csv")


if view == "Executive Dashboard":
    render_executive_dashboard()
elif view == "Company Profile":
    render_company_profile()
elif view == "Methodology":
    render_methodology()
elif view == "Data Sources":
    render_sources()
else:
    render_executive_dashboard()
