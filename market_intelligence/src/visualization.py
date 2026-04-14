# market_intelligence/src/visualization.py
#
# Produces three publication-ready figures and one summary table:
#
#   Figure 1 — opportunity_ranking.png
#       Horizontal stacked bar chart. Each bar = one market's composite score,
#       decomposed into weighted pillar contributions. Shows both the headline
#       ranking and the mix of factors driving each market's score.
#
#   Figure 2 — pillar_heatmap.png
#       Markets × Pillars heatmap. Lets a reviewer instantly spot where a market
#       is strong or weak across dimensions — useful for screening conversations.
#
#   Figure 3 — pillar_radar.png
#       Overlapping radar/spider chart comparing up to 4 markets across pillars.
#       Good for side-by-side comparison in a presentation context.
#
#   Table  — market_summary.csv  (clean, export-ready)
#       Top-N markets with composite score, pillar scores, rank, and ISO/RTO.
#
# All functions return matplotlib Figure objects so callers control where they
# are saved. generate_all() is the single entry point that saves everything.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in scripts and notebooks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Style constants ───────────────────────────────────────────────────────────
# Muted, institutional palette — four pillars, four distinct but low-saturation
# colors. Chosen to be readable in grayscale and on projected slides.

PILLAR_COLORS: dict[str, str] = {
    "demand":      "#1f4e79",   # deep navy — demand / commercial
    "energy":      "#1e6348",   # deep green — energy / clean power
    "feasibility": "#7b3f20",   # dark rust — execution / deployment
    "strategic":   "#4a3273",   # deep purple — strategic fit
}

PILLAR_LABELS: dict[str, str] = {
    "demand":      "Demand\nPotential",
    "energy":      "Energy\nAttractiveness",
    "feasibility": "Deployment\nFeasibility",
    "strategic":   "Strategic\nFit",
}

# Global figure styling applied at the module level.
# Using a clean base style and overriding the handful of defaults we care about.
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e5e5",
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

FIGURE_WIDTH  = 11   # inches — fits a standard slide or A4 landscape
FIGURE_HEIGHT = 6    # inches


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pillar_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of pillar score columns present in the DataFrame."""
    expected = [f"{p}_score" for p in PILLAR_COLORS.keys()]
    return [c for c in expected if c in df.columns]


def _pillar_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose each market's composite opportunity_score into the weighted
    contribution from each pillar.

    contribution_i = pillar_score_i * pillar_weight_i

    The sum of contributions equals the composite score, so stacking them
    produces a bar whose total length is the opportunity_score.
    """
    from market_intelligence.src.config import PILLAR_WEIGHTS

    contribs = pd.DataFrame(index=df.index)
    for pillar, weight in PILLAR_WEIGHTS.items():
        col = f"{pillar}_score"
        if col in df.columns:
            contribs[pillar] = df[col] * weight
    return contribs


def _short_name(market_name: str) -> str:
    """Shorten a market name for axis labels (keeps charts readable)."""
    replacements = {
        "ERCOT Texas":           "ERCOT (TX)",
        "PJM Northern Virginia": "PJM – N. Virginia",
        "PJM Ohio":              "PJM – Ohio",
        "MISO Iowa":             "MISO (IA)",
        "SPP Kansas":            "SPP (KS)",
        "CAISO California":      "CAISO (CA)",
        "Pacific Northwest":     "Pacific NW (WA)",
        "Southeast Georgia":     "Southeast (GA)",
    }
    return replacements.get(market_name, market_name)


# ── Figure 1: Ranked stacked bar chart ───────────────────────────────────────

def plot_opportunity_ranking(
    scored_df: pd.DataFrame,
    title: str = "Market Opportunity Score — Ranked by Composite",
    subtitle: str = "Bars decomposed into weighted pillar contributions  |  Score range: 0–100",
) -> plt.Figure:
    """
    Horizontal stacked bar chart of composite opportunity scores.

    Markets are sorted highest → lowest (best opportunity at the top).
    Each bar is split into pillar contributions so the driver of a market's
    ranking is immediately visible — this is the single most useful chart
    for an executive or BD conversation.
    """
    df = scored_df.copy().sort_values("opportunity_score", ascending=True)
    df["label"] = df["market_name"].apply(_short_name)

    contributions = _pillar_contributions(df)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.yaxis.grid(False)

    # Plot each pillar as a stacked segment
    left = np.zeros(len(df))
    for pillar, color in PILLAR_COLORS.items():
        if pillar not in contributions.columns:
            continue
        values = contributions[pillar].values
        bars = ax.barh(
            df["label"],
            values,
            left=left,
            color=color,
            label=PILLAR_LABELS[pillar].replace("\n", " "),
            height=0.55,
            edgecolor="white",
            linewidth=0.4,
        )
        left += values

    # Annotate the composite score at the right end of each bar
    for i, (_, row) in enumerate(df.iterrows()):
        score = row["opportunity_score"]
        rank  = int(row["rank"])
        ax.text(
            score + 0.6,
            i,
            f"{score:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
            fontweight="semibold",
        )
        # Rank label on the left
        ax.text(
            -1.5,
            i,
            f"#{rank}",
            va="center",
            ha="right",
            fontsize=8.5,
            color="#888888",
        )

    # Axis formatting
    ax.set_xlim(-4, 105)
    ax.set_xlabel("Opportunity Score (0–100)", labelpad=8, color="#444444")
    ax.tick_params(axis="y", labelsize=10, colors="#333333")
    ax.tick_params(axis="x", labelsize=9, colors="#666666")
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Legend — placed outside the chart area to the right
    legend = ax.legend(
        title="Pillar",
        title_fontsize=9,
        fontsize=8.5,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    legend.get_title().set_color("#444444")

    # Title block
    fig.text(0.08, 1.01, title, fontsize=13, fontweight="bold", color="#1a1a1a",
             transform=ax.transAxes)
    fig.text(0.08, 0.97, subtitle, fontsize=8.5, color="#777777",
             transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ── Figure 2: Pillar heatmap ──────────────────────────────────────────────────

def plot_pillar_heatmap(
    scored_df: pd.DataFrame,
    title: str = "Pillar Score Heatmap — Strengths and Gaps by Market",
) -> plt.Figure:
    """
    Heatmap of pillar scores (markets × pillars).

    Darker cells = higher score (more attractive).
    Useful for quickly identifying where each market excels or falls short
    without looking at individual metrics. A good complement to the bar chart.
    """
    df = scored_df.copy().sort_values("opportunity_score", ascending=False)
    df["label"] = df["market_name"].apply(_short_name)

    pillar_order = list(PILLAR_COLORS.keys())
    pillar_display = [PILLAR_LABELS[p] for p in pillar_order]

    # Build matrix: rows = markets (best at top), cols = pillars
    matrix = df[[f"{p}_score" for p in pillar_order]].values

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Use a clean sequential colormap — avoids red/green colorblind issues
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)

    # Annotate each cell with the numeric score
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val > 62 else "#333333"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=10, fontweight="semibold", color=text_color)

    # Axis labels
    ax.set_xticks(range(len(pillar_order)))
    ax.set_xticklabels(pillar_display, fontsize=9.5, color="#444444")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"#{int(r['rank'])}  {_short_name(r['market_name'])}" for _, r in df.iterrows()],
        fontsize=9.5,
        color="#333333",
    )
    ax.tick_params(length=0)  # remove tick marks — cell borders are enough

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Pillar Score (0–100)", fontsize=8.5, color="#555555")
    cbar.ax.tick_params(labelsize=8, colors="#666666")
    cbar.outline.set_edgecolor("#cccccc")

    # Remove all spines — the heatmap cells provide structure
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.grid(False)

    fig.text(0.02, 1.02, title, fontsize=12, fontweight="bold", color="#1a1a1a",
             transform=ax.transAxes)
    fig.text(0.02, 0.98, "Score range 0–100  |  Higher = more attractive",
             fontsize=8.5, color="#777777", transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ── Figure 3: Radar / spider chart ───────────────────────────────────────────

def plot_pillar_radar(
    scored_df: pd.DataFrame,
    market_ids: Optional[list[str]] = None,
    title: str = "Pillar Profile — Market Comparison",
) -> plt.Figure:
    """
    Radar chart comparing up to 4 markets across the four pillars.

    If market_ids is None, defaults to the top 4 by opportunity_score.
    Radar charts are appropriate here because the four pillars are conceptually
    equal-radius axes — we are comparing shapes, not a single dimension.
    """
    df = scored_df.copy()

    if market_ids is None:
        # Default: top 4 by rank
        market_ids = (
            df.sort_values("opportunity_score", ascending=False)
            .head(4)["market_id"]
            .tolist()
        )

    df_sel = df[df["market_id"].isin(market_ids)].copy()
    if df_sel.empty:
        raise ValueError(f"None of the requested market_ids found: {market_ids}")

    pillar_order  = list(PILLAR_COLORS.keys())
    pillar_labels = [PILLAR_LABELS[p] for p in pillar_order]
    n_pillars     = len(pillar_order)

    # Compute evenly-spaced angles for each axis (close the polygon by repeating first)
    angles = np.linspace(0, 2 * np.pi, n_pillars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})

    # Gridline and spoke styling
    ax.set_facecolor("#fafafa")
    ax.set_theta_offset(np.pi / 2)   # start at 12 o'clock
    ax.set_theta_direction(-1)        # clockwise
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7.5, color="#aaaaaa")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.spines["polar"].set_color("#cccccc")

    # Axis labels (pillar names)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pillar_labels, fontsize=9.5, color="#444444")

    # One line per market
    line_colors = ["#1f4e79", "#c0392b", "#1e6348", "#6c3483"]
    for i, (_, row) in enumerate(df_sel.iterrows()):
        values = [row[f"{p}_score"] for p in pillar_order]
        values += values[:1]  # close the polygon

        color = line_colors[i % len(line_colors)]
        ax.plot(angles, values, color=color, linewidth=1.8, linestyle="solid", zorder=3)
        ax.fill(angles, values, color=color, alpha=0.08)

        # Label the highest point on each market's polygon
        max_idx = np.argmax(values[:-1])
        ax.annotate(
            _short_name(row["market_name"]),
            xy=(angles[max_idx], values[max_idx]),
            fontsize=8,
            color=color,
            fontweight="semibold",
            ha="center",
        )

    # Legend
    handles = [
        plt.Line2D([0], [0], color=line_colors[i], linewidth=2)
        for i in range(len(df_sel))
    ]
    labels = [_short_name(r["market_name"]) for _, r in df_sel.iterrows()]
    ax.legend(handles, labels, loc="upper right",
              bbox_to_anchor=(1.35, 1.12), fontsize=8.5,
              frameon=True, framealpha=0.95, edgecolor="#cccccc")

    fig.suptitle(title, fontsize=12, fontweight="bold", color="#1a1a1a", y=1.02)
    fig.text(0.5, -0.01, "Pillar scores 0–100  |  Larger polygon = higher opportunity",
             ha="center", fontsize=8.5, color="#777777")

    fig.tight_layout()
    return fig


# ── Table export ──────────────────────────────────────────────────────────────

def export_summary_table(
    scored_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Export a clean summary CSV: rank, market, ISO, composite score, pillar scores.

    Rounds floats to one decimal place for readability.
    Returns the table as a DataFrame so it can also be displayed inline.
    """
    pillar_cols = _pillar_cols(scored_df)

    keep_cols = (
        ["rank", "market_id", "market_name", "iso_rto", "primary_states",
         "opportunity_score"]
        + pillar_cols
    )
    keep_cols = [c for c in keep_cols if c in scored_df.columns]

    table = (
        scored_df[keep_cols]
        .sort_values("rank")
        .head(top_n)
        .copy()
    )

    # Rename pillar columns to be more readable in the exported file
    rename = {f"{p}_score": f"{p.title()} Score" for p in PILLAR_COLORS.keys()}
    rename["opportunity_score"] = "Opportunity Score"
    table = table.rename(columns=rename)

    # Round all float columns to 1 decimal
    float_cols = table.select_dtypes("float").columns
    table[float_cols] = table[float_cols].round(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


# ── Full results export ───────────────────────────────────────────────────────

_QUADRANT_TO_RECOMMENDATION: dict[str, str] = {
    "High Opportunity / Low Risk":  "Priority",
    "High Opportunity / High Risk": "Proceed with Caution",
    "Low Opportunity / Low Risk":   "Monitor",
    "Low Opportunity / High Risk":  "Avoid",
}


def export_full_results(
    scored_df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Export a comprehensive single-file CSV with all key analytical outputs.

    Columns included:
        rank, market_id, market_name, iso_rto, primary_states,
        opportunity_score, [four pillar scores],
        aggregate_risk, opportunity_risk_quadrant, recommendation

    The recommendation column maps the 2×2 quadrant classification to an
    action label: Priority / Proceed with Caution / Monitor / Avoid.

    This is the intended starting point for any downstream deck or report —
    one row per market, all scored dimensions in one place.
    """
    from market_intelligence.src.risk_assessment import flag_opportunity_risk_quadrants

    # Merge in aggregate_risk and opportunity_risk_quadrant if not already present
    if "aggregate_risk" not in scored_df.columns:
        result = flag_opportunity_risk_quadrants(scored_df)
    else:
        result = scored_df.copy()

    if "opportunity_risk_quadrant" in result.columns:
        result["recommendation"] = result["opportunity_risk_quadrant"].map(
            _QUADRANT_TO_RECOMMENDATION
        )

    pillar_cols = _pillar_cols(result)
    extra_cols  = [c for c in ("aggregate_risk", "opportunity_risk_quadrant", "recommendation")
                   if c in result.columns]
    keep_cols   = (
        ["rank", "market_id", "market_name", "iso_rto", "primary_states",
         "opportunity_score"]
        + pillar_cols
        + extra_cols
    )
    keep_cols = [c for c in keep_cols if c in result.columns]

    table = result[keep_cols].sort_values("rank").copy()

    rename = {f"{p}_score": f"{p.title()} Score" for p in PILLAR_COLORS.keys()}
    rename["opportunity_score"] = "Opportunity Score"
    table = table.rename(columns=rename)

    float_cols = table.select_dtypes("float").columns
    table[float_cols] = table[float_cols].round(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


# ── Master entry point ────────────────────────────────────────────────────────

def generate_all(
    scored_df: pd.DataFrame,
    figures_dir: str | Path = "market_intelligence/outputs/figures",
    tables_dir:  str | Path = "market_intelligence/outputs/tables",
    radar_market_ids: Optional[list[str]] = None,
) -> dict[str, Path]:
    """
    Generate and save all figures and tables from a scored market DataFrame.

    Returns a dict mapping output name → saved Path, so callers can reference
    or display outputs programmatically.

    Usage:
        from market_intelligence.src.visualization import generate_all
        paths = generate_all(scored_df)
    """
    figures_dir = Path(figures_dir)
    tables_dir  = Path(tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # Figure 1 — Ranked stacked bar
    print("[VIZ] Generating figure 1: opportunity ranking...")
    fig1 = plot_opportunity_ranking(scored_df)
    p1 = figures_dir / "opportunity_ranking.png"
    fig1.savefig(p1)
    plt.close(fig1)
    saved["opportunity_ranking"] = p1
    print(f"      Saved: {p1}")

    # Figure 2 — Pillar heatmap
    print("[VIZ] Generating figure 2: pillar heatmap...")
    fig2 = plot_pillar_heatmap(scored_df)
    p2 = figures_dir / "pillar_heatmap.png"
    fig2.savefig(p2)
    plt.close(fig2)
    saved["pillar_heatmap"] = p2
    print(f"      Saved: {p2}")

    # Figure 3 — Radar chart (top 4 markets by default)
    print("[VIZ] Generating figure 3: pillar radar...")
    fig3 = plot_pillar_radar(scored_df, market_ids=radar_market_ids)
    p3 = figures_dir / "pillar_radar.png"
    fig3.savefig(p3)
    plt.close(fig3)
    saved["pillar_radar"] = p3
    print(f"      Saved: {p3}")

    # Table — Summary CSV
    print("[VIZ] Generating summary table...")
    tp = tables_dir / "market_summary.csv"
    table = export_summary_table(scored_df, tp)
    saved["market_summary"] = tp
    print(f"      Saved: {tp}")
    print(f"\n{table.to_string(index=False)}\n")

    return saved
