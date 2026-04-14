# market_intelligence/src/visualization_extended.py
#
# Additional figures for Wave 6 outputs.
# Three new charts:
#
#   Figure 4 — archetype_rankings.png
#       Grouped bar chart showing how each market's rank shifts across
#       the three customer archetypes vs. the baseline. Markets that
#       move significantly are archetype-specific opportunities.
#
#   Figure 5 — scenario_robustness.png
#       Slope chart (bump chart) showing rank position across all four
#       scenarios. Markets with flat lines are robust; markets with
#       steep slopes are sensitive to the structural assumption.
#
#   Figure 6 — opportunity_risk_matrix.png
#       2×2 scatter: x-axis = opportunity score, y-axis = aggregate risk
#       (encoded as 0/1/2). Each market is labeled. Quadrant lines at the
#       median opportunity score. The clearest "pursue" and "avoid" signals
#       are in the top-right and bottom-left quadrants respectively.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import numpy as np
import pandas as pd

from market_intelligence.src.visualization import (
    PILLAR_COLORS, FIGURE_WIDTH, FIGURE_HEIGHT, _short_name
)
from market_intelligence.src.customer_targeting import (
    archetype_ranking_comparison, ARCHETYPES
)
from market_intelligence.src.scenarios import scenario_rank_comparison, SCENARIOS
from market_intelligence.src.risk_assessment import (
    flag_opportunity_risk_quadrants, risk_matrix
)


# ── Figure 4: Archetype ranking comparison ────────────────────────────────────

def plot_archetype_rankings(
    metric_scores: pd.DataFrame,
    raw_df: pd.DataFrame,
    title: str = "Market Rankings by Customer Archetype",
) -> plt.Figure:
    """
    Grouped horizontal bar chart: one bar per archetype per market.

    Shows rank (1 = best) rather than score so the chart is easy to read
    without knowing the scale. Markets are sorted by baseline rank.
    Rank bars are colored by archetype.
    """
    comparison = archetype_ranking_comparison(metric_scores, raw_df)
    comparison["label"] = comparison["market_name"].apply(_short_name)

    archetype_ids    = list(ARCHETYPES.keys())
    archetype_labels = [ARCHETYPES[a]["label"].split("(")[0].strip() for a in archetype_ids]
    archetype_colors = ["#1f4e79", "#1e6348", "#7b3f20"]  # navy, green, rust

    n_markets     = len(comparison)
    n_archetypes  = len(archetype_ids)
    bar_height    = 0.22
    group_spacing = 1.0
    y_positions   = np.arange(n_markets) * group_spacing

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(6, n_markets * 0.9)))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.yaxis.grid(False)

    # Invert rank so that rank 1 = longest bar (most intuitive)
    max_rank = n_markets

    for j, (arch_id, color, label) in enumerate(
        zip(archetype_ids, archetype_colors, archetype_labels)
    ):
        col = f"{arch_id}_rank"
        if col not in comparison.columns:
            continue

        offsets    = y_positions + (j - n_archetypes / 2 + 0.5) * bar_height
        rank_vals  = comparison[col].values
        bar_vals   = max_rank + 1 - rank_vals  # invert: rank 1 → tallest

        bars = ax.barh(
            offsets,
            bar_vals,
            height=bar_height * 0.85,
            color=color,
            alpha=0.85,
            label=label,
            edgecolor="white",
            linewidth=0.3,
        )

        # Annotate with rank number at bar end
        for offset, rank in zip(offsets, rank_vals):
            ax.text(
                bar_vals[list(offsets).index(offset)] + 0.05,
                offset,
                f"#{int(rank)}",
                va="center", ha="left",
                fontsize=7.5, color=color,
            )

    # Market labels on y-axis (centered on group)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(comparison["label"], fontsize=9.5, color="#333333")

    # X-axis: show as rank labels
    ax.set_xlim(0, max_rank + 1.5)
    tick_positions = max_rank + 1 - np.arange(1, max_rank + 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"#{i}" for i in range(1, max_rank + 1)],
                       fontsize=8, color="#666666")
    ax.set_xlabel("Market Rank (leftmost = #1 best for that archetype)",
                  labelpad=8, color="#444444")

    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    ax.legend(
        title="Customer Archetype",
        title_fontsize=9,
        fontsize=8.5,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    fig.text(0.06, 1.02, title, fontsize=12, fontweight="bold",
             color="#1a1a1a", transform=ax.transAxes)
    fig.text(0.06, 0.98,
             "Bar length = rank position  |  Longer = higher-ranked for that archetype",
             fontsize=8.5, color="#777777", transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ── Figure 5: Scenario robustness slope chart ─────────────────────────────────

def plot_scenario_robustness(
    raw_df: pd.DataFrame,
    title: str = "Ranking Stability Across Scenarios",
) -> plt.Figure:
    """
    Slope / bump chart: each column = one scenario, each line = one market.
    Y-axis = rank (1 at top).

    Flat lines = robust markets. Steep lines = scenario-sensitive markets.
    This is the clearest way to show which markets are consistent bets
    vs. which depend on specific structural assumptions being correct.
    """
    comparison  = scenario_rank_comparison(raw_df)
    scenario_ids = [s for s in SCENARIOS.keys()]
    scenario_labels = [SCENARIOS[s]["label"] for s in scenario_ids]
    rank_cols   = [f"{s}_rank" for s in scenario_ids]

    n_scenarios = len(scenario_ids)
    n_markets   = len(comparison)
    x_positions = np.arange(n_scenarios)

    # Color scale: top-ranked markets get darker lines, bottom-ranked lighter
    baseline_ranks = comparison["baseline_rank"].values
    cmap  = matplotlib.colormaps["Blues_r"].resampled(n_markets + 2)
    colors = [cmap(r / (n_markets + 1)) for r in baseline_ranks]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT + 1))
    ax.invert_yaxis()  # rank 1 at top
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(comparison.iterrows()):
        ranks = [row[col] for col in rank_cols]
        color = colors[i]
        label = _short_name(row["market_name"])

        ax.plot(x_positions, ranks, color=color, linewidth=2.0,
                marker="o", markersize=6, zorder=3)

        # Label at the leftmost point (baseline)
        ax.text(
            -0.12, ranks[0],
            label,
            va="center", ha="right",
            fontsize=8.5, color=color,
            fontweight="semibold" if ranks[0] <= 3 else "normal",
        )
        # Label at the rightmost point
        ax.text(
            n_scenarios - 1 + 0.08, ranks[-1],
            f"#{int(ranks[-1])}",
            va="center", ha="left",
            fontsize=8, color=color,
        )

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        scenario_labels,
        fontsize=9,
        color="#444444",
        wrap=True,
    )
    ax.set_yticks(range(1, n_markets + 1))
    ax.set_yticklabels([f"Rank {i}" for i in range(1, n_markets + 1)],
                       fontsize=8.5, color="#555555")
    ax.set_xlim(-1.2, n_scenarios + 0.5)
    ax.set_ylim(n_markets + 0.5, 0.5)

    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["right"].set_visible(False)

    fig.text(0.06, 1.02, title, fontsize=12, fontweight="bold",
             color="#1a1a1a", transform=ax.transAxes)
    fig.text(0.06, 0.98,
             "Flat lines = robust across scenarios  |  Steep lines = scenario-sensitive",
             fontsize=8.5, color="#777777", transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ── Figure 6: Opportunity vs. risk scatter ────────────────────────────────────

def plot_opportunity_risk_matrix(
    scored_df: pd.DataFrame,
    title: str = "Opportunity Score vs. Aggregate Risk",
) -> plt.Figure:
    """
    Scatter plot: x = opportunity_score, y = aggregate risk (numeric).

    Quadrant lines at median opportunity score and Medium risk level.
    Each market is labeled. Color encodes risk level.
    """
    _RISK_NUM = {"Low": 0, "Medium": 1, "High": 2}
    _RISK_COLORS = {"Low": "#1e6348", "Medium": "#e67e22", "High": "#c0392b"}

    result = flag_opportunity_risk_quadrants(scored_df)
    result["risk_num"] = result["aggregate_risk"].map(_RISK_NUM)

    # Compute deterministic vertical jitter so markets at the same risk level
    # don't overlap. Within each risk group, distribute evenly over ±0.12 units.
    result = result.reset_index(drop=True)
    jitter_vals = np.zeros(len(result))
    spread = 0.24
    for risk_val in result["risk_num"].unique():
        mask = result["risk_num"] == risk_val
        idxs = result[mask].index.tolist()
        n = len(idxs)
        if n == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-spread / 2, spread / 2, n).tolist()
        for i, idx in enumerate(idxs):
            jitter_vals[idx] = offsets[i]
    result["jitter"] = jitter_vals

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_axisbelow(True)
    ax.grid(True, color="#e5e5e5", linewidth=0.5)

    # Quadrant dividers
    median_score = result["opportunity_score"].median()
    ax.axvline(x=median_score, color="#cccccc", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(y=1.0, color="#cccccc", linewidth=1.2, linestyle="--", zorder=1)  # between Low and High

    # Quadrant labels (faint background text)
    for (x_pos, y_pos, text) in [
        (median_score + 1,  1.7, "High Opportunity\nHigh Risk"),
        (median_score - 1,  1.7, "Low Opportunity\nHigh Risk"),
        (median_score + 1,  0.2, "High Opportunity\nLow Risk"),
        (median_score - 1,  0.2, "Low Opportunity\nLow Risk"),
    ]:
        ha = "left" if x_pos > median_score else "right"
        ax.text(x_pos, y_pos, text, ha=ha, va="center",
                fontsize=8, color="#cccccc", style="italic")

    # Plot each market
    for _, row in result.iterrows():
        color = _RISK_COLORS[row["aggregate_risk"]]
        y_pos = row["risk_num"] + row["jitter"]
        ax.scatter(
            row["opportunity_score"],
            y_pos,
            s=120,
            color=color,
            zorder=4,
            edgecolors="white",
            linewidth=0.8,
        )
        ax.annotate(
            _short_name(row["market_name"]),
            xy=(row["opportunity_score"], y_pos),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight="semibold",
        )

    # Axes
    ax.set_xlabel("Opportunity Score (0–100)", labelpad=8, color="#444444")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low Risk", "Medium Risk", "High Risk"],
                       fontsize=9.5, color="#444444")
    ax.set_xlim(30, 80)
    ax.set_ylim(-0.5, 2.5)

    # Legend
    handles = [
        mpatches.Patch(color=c, label=label)
        for label, c in _RISK_COLORS.items()
    ]
    ax.legend(handles=handles, title="Aggregate Risk",
              title_fontsize=9, fontsize=8.5,
              loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    fig.text(0.08, 1.02, title, fontsize=12, fontweight="bold",
             color="#1a1a1a", transform=ax.transAxes)
    fig.text(0.08, 0.98,
             "Top-right quadrant = highest-priority diligence targets  |  "
             "Bottom-right = most actionable",
             fontsize=8.5, color="#777777", transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ── Master entry point ────────────────────────────────────────────────────────

def generate_extended_outputs(
    scored_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    metric_scores: pd.DataFrame,
    figures_dir: str | Path = "market_intelligence/outputs/figures",
    tables_dir:  str | Path = "market_intelligence/outputs/tables",
) -> dict[str, Path]:
    """
    Generate and save the three Wave 6 figures.
    Returns a dict of {output_name: saved_path}.
    """
    figures_dir = Path(figures_dir)
    tables_dir  = Path(tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    print("[VIZ] Generating figure 4: archetype rankings...")
    fig4 = plot_archetype_rankings(metric_scores, raw_df)
    p4 = figures_dir / "archetype_rankings.png"
    fig4.savefig(p4)
    plt.close(fig4)
    saved["archetype_rankings"] = p4
    print(f"      Saved: {p4}")

    print("[VIZ] Generating figure 5: scenario robustness slope chart...")
    fig5 = plot_scenario_robustness(raw_df)
    p5 = figures_dir / "scenario_robustness.png"
    fig5.savefig(p5)
    plt.close(fig5)
    saved["scenario_robustness"] = p5
    print(f"      Saved: {p5}")

    print("[VIZ] Generating figure 6: opportunity vs. risk matrix...")
    fig6 = plot_opportunity_risk_matrix(scored_df)
    p6 = figures_dir / "opportunity_risk_matrix.png"
    fig6.savefig(p6)
    plt.close(fig6)
    saved["opportunity_risk_matrix"] = p6
    print(f"      Saved: {p6}")

    return saved
