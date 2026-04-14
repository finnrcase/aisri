# Market Opportunity Engine — AI Infrastructure Siting

A decision-support tool for screening and ranking U.S. power markets for new
AI-coupled energy infrastructure development. Built as part of the AISRI project.

---

## What it does

This module scores eight U.S. wholesale power markets across four pillars —
Demand Potential, Energy Attractiveness, Deployment Feasibility, and Strategic
Fit — and produces a ranked composite opportunity score for each market.

The output is designed to support early-stage market prioritization: the kind
of structured screening a business development or strategy team would run before
committing time to a full market entry feasibility study.

---

## Why it was built

New AI infrastructure projects — data centers, AI campuses, co-located
generation and compute — require developers to make multi-year commitments to
specific geographies before a single megawatt of load is operational. Choosing
the wrong market has significant capital and time costs.

Most market selection decisions at this stage are made informally, based on
where demand already exists or where a company has existing relationships.
This tool makes the selection logic explicit, auditable, and updateable —
which is useful both for internal alignment and for communicating a market
thesis to investors or partners.

It is not a financial model. It is a screening layer that identifies which
markets are worth investing time in before building one.

---

## Framework

Four pillars, 13 metrics, one composite opportunity score (0–100, higher = better).

| Pillar | Weight | What it captures |
|---|---|---|
| Demand Potential | 30% | Hyperscale presence, installed capacity, tech employment, fiber connectivity |
| Energy Attractiveness | 30% | Wholesale power price, renewable capacity factor, clean energy penetration, grid reliability |
| Deployment Feasibility | 25% | Interconnection queue time, permitting speed, land cost, water availability |
| Strategic Fit | 15% | Renewable co-location potential, PPA market depth, state policy environment |

Normalization uses direction-aware min-max scaling with fixed bounds defined
in `src/config.py`. "Lower is better" metrics (e.g., power price, queue months)
are inverted before scaling so that a higher score always means a more attractive
market on that dimension.

All weights, bounds, and metric definitions are in a single configuration file
and can be adjusted without touching the scoring code.

---

## Markets covered (v1)

| Market | ISO/RTO | Notes |
|---|---|---|
| ERCOT Texas | ERCOT | Fastest permitting, most active PPA market, strong wind/solar pipeline |
| PJM Northern Virginia | PJM | Largest existing data center market; feasibility-constrained |
| PJM Ohio | PJM | Emerging hub with improving interconnection outlook |
| MISO Iowa | MISO | Highest wind capacity factors in CONUS; limited existing demand |
| SPP Kansas | SPP | Best renewable resource quality; near-greenfield demand story |
| CAISO California | CAISO | Strong policy and PPA market; most constrained development environment |
| Pacific Northwest | NWPP | Hydro-dominant baseload; best clean energy penetration in dataset |
| Southeast Georgia | SERC | Reliable regulated utility service; growing AI corridor |

---

## Usage

```bash
# From project root — runs the full scoring pipeline
python market_intelligence/run_market_score.py

# Interactive analysis and figures
jupyter notebook notebooks/market_opportunity_analysis.ipynb
```

**Dependencies:** `pandas`, `numpy`, `matplotlib` (see project `requirements.txt`)

---

## Outputs

| File | Description |
|---|---|
| `data/processed/market_scores.csv` | Full scored table: composite score, pillar scores, all 13 metric scores |
| `outputs/figures/opportunity_ranking.png` | Stacked bar chart of composite scores decomposed by pillar contribution |
| `outputs/figures/pillar_heatmap.png` | Markets × pillars heatmap of pillar scores |
| `outputs/figures/pillar_radar.png` | Radar chart comparing top-4 market profiles |
| `outputs/tables/market_summary.csv` | Clean summary table (rank, composite, pillar scores) |

---

## Caveats and limitations

**Data quality.** All values are populated from public sources (EIA, NREL, LBNL,
CBRE, NERC) and represent approximate 2023–2024 conditions. Several metrics use
ordinal scoring (1–5) where continuous data was not available from public sources.
This is appropriate for a screening tool but not for project-level analysis.

**Interconnection timelines** are median estimates based on published queue data.
Actual timelines are project-specific and depend on queue position, transmission
topology, and study outcomes that are not predictable at the screening stage.

**Wholesale power prices** reflect day-ahead spot price averages. The all-in
delivered cost of power for a large industrial customer — including transmission,
ancillary services, and any hedging — will differ, often materially.

**The weights reflect a specific perspective.** The baseline weights were set
from the perspective of a clean energy developer evaluating co-location
opportunities. A data center operator, a merchant generator, or a regulated
utility would likely weight the pillars differently. The sensitivity analysis
in the notebook tests two alternative weight schemes and finds the top 2–3 markets
are stable; the middle tier is more sensitive to weight assumptions.

**This is a v1 dataset covering 8 markets.** It is illustrative, not exhaustive.
A production version of this tool would include more markets, finer geographic
resolution (sub-market zones rather than state-level aggregates), and quarterly
data refreshes.

---

## File structure

```
market_intelligence/
├── data/
│   ├── raw/
│   │   └── market_inputs.csv          # Source dataset (one row per market)
│   └── processed/
│       └── market_scores.csv          # Pipeline output
├── src/
│   ├── config.py                      # All weights, bounds, metric definitions
│   ├── processing.py                  # Data loading and normalization
│   ├── scoring.py                     # Pillar and composite scoring
│   └── visualization.py              # Figure and table generation
├── outputs/
│   ├── figures/                       # Saved chart PNGs
│   └── tables/                        # Saved CSV summaries
├── docs/
│   ├── methodology_note.md            # Normalization and aggregation methodology
│   ├── variable_dictionary.md         # Full variable definitions with sources and limitations
│   ├── business_interpretation.md     # Market-by-market narrative and tradeoff analysis
│   ├── recommendation_framework.md   # Priority / Watchlist / Constrained classification
│   └── interview_guide.md             # 60s pitch, 2-min walkthrough, Q&A
├── run_market_score.py                # CLI entry point
└── README.md                          # This file
```
