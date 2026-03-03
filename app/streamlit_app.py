# app/streamlit_app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SCORES_PATH = DATA_DIR / "processed" / "scores_v1.csv"
METRICS_PATH = DATA_DIR / "processed" / "metrics_final.csv"
COMPANIES_PATH = DATA_DIR / "raw" / "extracted" / "companies.csv"
SOURCES_PATH = DATA_DIR / "raw" / "extracted" / "sources.csv"
METRIC_DEFS_PATH = DATA_DIR / "raw" / "extracted" / "metric_definitions.csv"


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="AISRI — AI Sustainability Risk Index",
    page_icon="🔵",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .small-muted { color: rgba(255,255,255,0.70); font-size: 0.9rem; }
      .card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px 16px;
        background: rgba(255,255,255,0.03);
      }
      .kpi {
        font-size: 1.4rem; font-weight: 700; margin: 0;
      }
      .kpi-label {
        font-size: 0.9rem; opacity: 0.75; margin: 0;
      }
      .pill {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04);
        font-size: 0.85rem; opacity: 0.95;
      }
      .hr { height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def _file_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    # mtime is included so cache invalidates when file changes
    return pd.read_csv(path)


def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not _exists(SCORES_PATH) or not _exists(METRICS_PATH):
        return None, None, None, None, None

    scores = _load_csv(str(SCORES_PATH), _file_mtime(SCORES_PATH))
    metrics = _load_csv(str(METRICS_PATH), _file_mtime(METRICS_PATH))

    companies = _load_csv(str(COMPANIES_PATH), _file_mtime(COMPANIES_PATH)) if _exists(COMPANIES_PATH) else None
    sources = _load_csv(str(SOURCES_PATH), _file_mtime(SOURCES_PATH)) if _exists(SOURCES_PATH) else None
    metric_defs = _load_csv(str(METRIC_DEFS_PATH), _file_mtime(METRIC_DEFS_PATH)) if _exists(METRIC_DEFS_PATH) else None

    return scores, metrics, companies, sources, metric_defs


def grade_badge(g: str) -> str:
    g = str(g).strip().upper()
    if g not in {"A", "B", "C", "D"}:
        return f"<span class='pill'>Grade: {g}</span>"
    return f"<span class='pill'>Confidence: {g}</span>"


def fmt_num(x, nd=2) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def fmt_pct(x, nd=0) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "—"


def fmt_ratio(x, nd=2) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def fmt_intensity_tco2_per_mwh(x, nd=5) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f} tCO₂/MWh"
    except Exception:
        return "—"


def safe_col(df: pd.DataFrame, col: str, default=np.nan):
    return df[col] if col in df.columns else default


# -----------------------------
# Header
# -----------------------------
st.markdown("## 🔵 AISRI — AI Sustainability Risk Index")
st.markdown(
    "<div class='small-muted'>A transparent, source-backed sustainability risk index for AI infrastructure. "
    "Scores are comparative and based on disclosed metrics and disclosure quality.</div>",
    unsafe_allow_html=True,
)

scores, metrics, companies, sources, metric_defs = load_data()

if scores is None or metrics is None:
    st.error("Missing required files. Run:\n\n- `python src/build_metrics_final.py`\n- `python src/run_score_v1.py`\n\nThen re-open the app.")
    st.stop()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("### Controls")
available_years = sorted(scores["fiscal_year"].dropna().unique().tolist()) if "fiscal_year" in scores.columns else [2024]
year = st.sidebar.selectbox("Fiscal year", available_years, index=len(available_years) - 1)

view = st.sidebar.radio("View", ["Rankings", "Company Profile", "Methodology", "Data Sources"], index=0)

search = st.sidebar.text_input("Search company_id / name", value="").strip().lower()

# If companies.csv exists, map company_id -> name
name_map = {}
if companies is not None and "company_id" in companies.columns and "company_name" in companies.columns:
    name_map = dict(zip(companies["company_id"].astype(str), companies["company_name"].astype(str)))


# -----------------------------
# Filter scores for year
# -----------------------------
scores_y = scores.copy()
if "fiscal_year" in scores_y.columns:
    scores_y = scores_y[scores_y["fiscal_year"] == year].copy()

# Add company_name for display
scores_y["company_name"] = scores_y["company_id"].astype(str).map(name_map).fillna(scores_y["company_id"].astype(str))

# Apply search filter
if search:
    mask = scores_y["company_id"].astype(str).str.lower().str.contains(search) | scores_y["company_name"].astype(str).str.lower().str.contains(search)
    scores_y = scores_y[mask].copy()

# Sort: lowest risk is "better"
if "overall_risk" in scores_y.columns:
    scores_y = scores_y.sort_values(["overall_risk", "confidence_score"], ascending=[True, False], na_position="last").reset_index(drop=True)
else:
    scores_y = scores_y.sort_values(["confidence_score"], ascending=[False], na_position="last").reset_index(drop=True)


# -----------------------------
# Rankings
# -----------------------------
def render_rankings():
    c1, c2, c3, c4 = st.columns(4)

    n_companies = len(scores_y)
    n_scored = int(scores_y["overall_risk"].notna().sum()) if "overall_risk" in scores_y.columns else 0
    avg_risk = float(scores_y["overall_risk"].mean()) if "overall_risk" in scores_y.columns and n_scored > 0 else np.nan
    avg_conf = float(scores_y["confidence_score"].mean()) if "confidence_score" in scores_y.columns else np.nan

    with c1:
        st.markdown("<div class='card'><p class='kpi'>{}</p><p class='kpi-label'>Companies (filtered)</p></div>".format(n_companies), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><p class='kpi'>{}</p><p class='kpi-label'>With risk score</p></div>".format(n_scored), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><p class='kpi'>{}</p><p class='kpi-label'>Avg risk (lower is better)</p></div>".format(fmt_num(avg_risk, 2)), unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><p class='kpi'>{}</p><p class='kpi-label'>Avg confidence</p></div>".format(fmt_num(avg_conf, 1)), unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Columns to show (only if present)
    show_cols = []
    for col in [
        "company_name",
        "company_id",
        "overall_risk",
        "confidence_score",
        "confidence_grade",
        "coverage",
        "pillar_energy",
        "pillar_efficiency",
        "pillar_carbon",
        "pillar_offsets",
        "pillar_transparency",
    ]:
        if col in scores_y.columns:
            show_cols.append(col)

    table = scores_y[show_cols].copy()

    # Pretty formatting in table
    if "overall_risk" in table.columns:
        table["overall_risk"] = table["overall_risk"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))
    if "confidence_score" in table.columns:
        table["confidence_score"] = table["confidence_score"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))
    if "coverage" in table.columns:
        table["coverage"] = table["coverage"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Note: Lower **overall_risk** is better. Missing values reflect missing required inputs (conservative null handling).")


# -----------------------------
# Company Profile
# -----------------------------
def render_company_profile():
    # Choose company from full year list (not filtered by search)
    scores_full_y = scores[scores["fiscal_year"] == year].copy() if "fiscal_year" in scores.columns else scores.copy()
    scores_full_y["company_name"] = scores_full_y["company_id"].astype(str).map(name_map).fillna(scores_full_y["company_id"].astype(str))

    options = scores_full_y.sort_values("company_name")["company_id"].astype(str).unique().tolist()
    default = options[0] if options else None

    chosen = st.selectbox("Select company", options, index=0 if default else None, format_func=lambda cid: f"{name_map.get(cid, cid)} ({cid})")

    if not chosen:
        st.info("No company selected.")
        return

    row = scores_full_y[scores_full_y["company_id"] == chosen].head(1)
    if row.empty:
        st.warning("No score row found for this company/year.")
        return

    r = row.iloc[0].to_dict()

    # Top summary cards
    a, b, c, d = st.columns(4)
    with a:
        st.markdown(
            f"<div class='card'><p class='kpi'>{fmt_num(r.get('overall_risk'),2)}</p><p class='kpi-label'>Overall risk</p></div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"<div class='card'><p class='kpi'>{fmt_num(r.get('confidence_score'),2)}</p><p class='kpi-label'>Confidence score</p></div>",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            f"<div class='card'><p class='kpi'>{fmt_num(r.get('coverage_score'),1) if 'coverage_score' in scores_full_y.columns else '—'}</p><p class='kpi-label'>Coverage score</p></div>",
            unsafe_allow_html=True,
        )
    with d:
        st.markdown(
            f"<div class='card'>{grade_badge(r.get('confidence_grade','—'))}<div style='height:8px'></div><p class='kpi-label'>Grade</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Pillar breakdown
    st.markdown("### Pillar breakdown")
    pill_cols = [c for c in ["pillar_energy", "pillar_efficiency", "pillar_carbon", "pillar_offsets", "pillar_transparency"] if c in scores_full_y.columns]
    if pill_cols:
        pillars = pd.DataFrame(
            {
                "pillar": [c.replace("pillar_", "").title() for c in pill_cols],
                "risk": [r.get(c) for c in pill_cols],
            }
        )
        pillars["risk"] = pillars["risk"].astype(float)
        st.bar_chart(pillars.set_index("pillar"))
    else:
        st.info("No pillar columns found in scores output.")

    st.markdown("### Underlying metrics (from metrics_final.csv)")
    m = metrics[(metrics["company_id"] == chosen) & (metrics["fiscal_year"] == year)].copy() if "fiscal_year" in metrics.columns else metrics[metrics["company_id"] == chosen].copy()
    if m.empty:
        st.warning("No metrics found for this company/year in metrics_final.csv.")
        return

    # Add metric labels if metric_defs exists
    if metric_defs is not None and "metric_id" in metric_defs.columns:
        label_map = {}
        if "label" in metric_defs.columns:
            label_map = dict(zip(metric_defs["metric_id"].astype(str), metric_defs["label"].astype(str)))
        m["metric_label"] = m["metric_id"].astype(str).map(label_map).fillna(m["metric_id"].astype(str))
    else:
        m["metric_label"] = m["metric_id"].astype(str)

    show = m[["metric_label", "metric_id", "value", "unit", "source_id", "extraction_note"]].copy()

    # Light formatting
    def pretty_value(row):
        mid = str(row["metric_id"])
        v = row["value"]
        if pd.isna(v):
            return "—"
        if mid in {"renewable_share_pct", "offset_share_scope2"}:
            return fmt_pct(v, 0)
        if mid == "pue":
            return fmt_ratio(v, 2)
        if mid == "scope2_intensity":
            return fmt_intensity_tco2_per_mwh(v, 5)
        if mid.startswith("reports_"):
            return "1" if float(v) == 1.0 else "0"
        if mid == "third_party_assurance_level":
            return str(int(float(v)))
        return fmt_num(v, 4)

    show["value"] = show.apply(pretty_value, axis=1)
    show = show.sort_values("metric_id")

    st.dataframe(show, use_container_width=True, hide_index=True)

    # History
    st.markdown("### History (if multiple years exist)")
    if "fiscal_year" in scores.columns:
        hist = scores[scores["company_id"] == chosen].copy()
        if hist["fiscal_year"].nunique() > 1 and "overall_risk" in hist.columns:
            hist = hist.sort_values("fiscal_year")
            line = hist[["fiscal_year", "overall_risk"]].set_index("fiscal_year")
            st.line_chart(line)
        else:
            st.caption("No multi-year history available yet for this company.")


# -----------------------------
# Methodology + Data Sources
# -----------------------------
def render_methodology():
    st.markdown("### Methodology (v1)")
    st.markdown(
        """
**AISRI** is a comparative sustainability risk index for AI companies. It does **not** estimate model-specific emissions.
It scores companies based on disclosed infrastructure sustainability metrics and disclosure quality.

**Pillars (typical v1):**
- **Energy sourcing** (renewable / carbon-free electricity share)
- **Efficiency** (PUE / data center efficiency)
- **Carbon exposure** (Scope 2 intensity / grid exposure)
- **Offsets / contractual instruments** (offset/REC reliance proxies when disclosed)
- **Transparency** (binary disclosure metrics + assurance)

**Conservative policy:** if a value cannot be confidently sourced, it is left **null**.
Nulls can reduce coverage / confidence and may trigger missing-data penalties (depending on scoring config).
        """
    )

    st.markdown("### Current outputs")
    st.write(
        {
            "scores_v1.csv": str(SCORES_PATH),
            "metrics_final.csv": str(METRICS_PATH),
            "companies.csv": str(COMPANIES_PATH) if _exists(COMPANIES_PATH) else "(missing)",
            "metric_definitions.csv": str(METRIC_DEFS_PATH) if _exists(METRIC_DEFS_PATH) else "(missing)",
        }
    )

    st.markdown("### Scoring inputs (this year)")
    st.dataframe(scores_y.head(25), use_container_width=True, hide_index=True)


def render_sources():
    st.markdown("### Data Sources")
    if sources is None or sources.empty:
        st.info("No sources.csv found (or it is empty). You can still run AISRI; sources improve traceability.")
        return

    s = sources.copy()
    # Common helpful columns (if they exist)
    keep = [c for c in ["source_id", "title", "url", "publisher", "year", "notes"] if c in s.columns]
    if keep:
        s = s[keep]
    st.dataframe(s.sort_values(keep[0]) if keep else s, use_container_width=True, hide_index=True)

    st.caption("Tip: keep source_id stable and reference it in metrics_raw/manual rows.")


# -----------------------------
# Render selected view
# -----------------------------
if view == "Rankings":
    render_rankings()
elif view == "Company Profile":
    render_company_profile()
elif view == "Methodology":
    render_methodology()
elif view == "Data Sources":
    render_sources()
else:
    render_rankings()