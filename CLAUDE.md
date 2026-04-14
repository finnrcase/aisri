# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**AISRI** (AI Sustainability Risk Index) scores AI companies on sustainability risk using disclosed metrics from their annual sustainability reports. Scores are 0–100 where **lower is better** (lower risk). The pipeline ends with a Streamlit dashboard.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (run in order):
python src/ingest_from_pdfs.py             # Extract metrics from PDFs -> data/raw/extracted/metrics_raw.csv
python src/build_metrics_final.py          # Merge extracted + manual overrides -> data/processed/metrics_final.csv
python src/run_score_v1.py                 # Score companies -> data/processed/scores_v1.csv

# Validate raw data CSVs
python src/validate_all.py

# Run Streamlit app
streamlit run app/streamlit_app.py

# Ingest a single company only
python src/ingest_from_pdfs.py --company google
python src/ingest_from_pdfs.py --company microsoft --pdf path/to/report.pdf
```

## Architecture

### Data flow

```
data/raw/pdfs/*.pdf
    └─> src/ingest_from_pdfs.py  ──> data/raw/extracted/metrics_raw.csv
                                                     │
data/raw/manual/metrics_manual.csv ─────────────────┤
                                                     ▼
                                    src/build_metrics_final.py
                                                     │
                                                     ▼
                                    data/processed/metrics_final.csv
                                                     │
                                    src/run_score_v1.py
                                                     │
                                                     ▼
                                    data/processed/scores_v1.csv
                                                     │
                                    app/streamlit_app.py (display)
```

### Key source files

- **[src/schema.py](src/schema.py)** — Pydantic row schemas for `metrics_raw.csv`, `companies.csv`, `sources.csv`. Also defines controlled vocabulary types (`Pillar`, `Direction`, `Scope`, `QualityFlag`, `Boundary`) and a `Codebook` loader that validates `metric_id` against `metric_definitions.csv`.

- **[src/normalize.py](src/normalize.py)** — `linear_risk()`, `binary_risk()`, `assurance_risk()`: all return 0–100 risk scores. `V1_BOUNDS` defines the min/max bounds used for linear normalization of continuous metrics.

- **[src/scoring.py](src/scoring.py)** — `compute_scores()`: aggregates per-metric risk into pillar scores, then into `overall_risk`. Pillar weights and metric weights are hardcoded here (v1). Missing transparency metrics default to 100 risk; other missing metrics default to `missing_risk_default` (65).

- **[src/confidence.py](src/confidence.py)** — `compute_confidence()`: produces a data quality score (0–100, grade A–D) based on coverage, source quality, recency, and assurance. Separate from the risk score.

- **[src/run_score_v1.py](src/run_score_v1.py)** — Orchestrates scoring + confidence, applies deterministic NaN handling (replaces NaN pillars with `MISSING_PENALTY = 65`), merges confidence columns, writes `scores_v1.csv`.

- **[src/build_metrics_final.py](src/build_metrics_final.py)** — Upserts `metrics_manual.csv` onto `metrics_raw.csv` by key `(company_id, metric_id, fiscal_year)`. Manual rows win on key conflict. Produces `metrics_final.csv`.

- **[src/ingest_from_pdfs.py](src/ingest_from_pdfs.py)** — PDF text extraction using `pypdf`. Each company has a dedicated extractor function (`extract_google_2024`, `extract_microsoft_2024`, `extract_amazon_2024`). Register new companies in the `EXTRACTORS` dict. Upserts results into `metrics_raw.csv`.

### Scoring design

Five pillars with v1 weights: `energy` 25%, `efficiency` 20%, `carbon` 25%, `offsets` 15%, `transparency` 15%.

Required metrics: `renewable_share_pct`, `pue`, `scope2_intensity`, `offset_share_scope2`, plus four binary transparency flags (`reports_scope2_market_and_location`, `reports_electricity_consumption`, `reports_data_center_metrics`, `third_party_assurance_level`).

`third_party_assurance_level` is coded 0/1/2 (none/limited/reasonable), not binary.

### Adding a new company

1. Add an extractor function in [src/ingest_from_pdfs.py](src/ingest_from_pdfs.py) returning `List[MetricRow]`.
2. Register it in `EXTRACTORS` and `DEFAULT_FILENAMES`.
3. Add a row for the company in `data/raw/extracted/companies.csv`.
4. Place the PDF in `data/raw/pdfs/` and run the pipeline.

### Manual overrides

Put corrections in `data/raw/manual/metrics_manual.csv` using the canonical columns. Manual rows take precedence over extracted rows on the same `(company_id, metric_id, fiscal_year)` key. Audit columns `page_ref` and `verbatim_snippet` are allowed here only.

---

## Market Intelligence module

A separate BD/strategy module that ranks U.S. power markets for AI infrastructure development. Scores are 0–100 where **higher is better** (higher opportunity) — the inverse of AISRI's risk convention. The output column is `opportunity_score`.

### Commands

```bash
# Run baseline scoring only
python market_intelligence/run_market_score.py

# Run full extended analysis (all four layers + all figures)
python market_intelligence/run_extended_analysis.py

# Interactive analysis notebook
jupyter notebook notebooks/market_opportunity_analysis.ipynb
```

### Data flow

```
market_intelligence/data/raw/market_inputs.csv
    └─> src/processing.py       (normalize_all)       ──> metric scores 0–100
    └─> src/scoring.py          (score_markets)        ──> pillar scores + composite
    └─> src/customer_targeting.py (score_for_archetype) ──> per-archetype rankings
    └─> src/scenarios.py        (run_all_scenarios)    ──> scenario rank comparison
    └─> src/risk_assessment.py  (risk_matrix)          ──> regulatory/grid/execution ratings
    └─> src/economics.py        (economics_summary)    ──> cost/revenue/payback table
    └─> src/visualization.py + visualization_extended.py ──> outputs/figures/*.png
                                                        └─> outputs/tables/*.csv
```

### Key source files

- **[market_intelligence/src/config.py](market_intelligence/src/config.py)** — Single source of truth for all weights and metric definitions. `PILLAR_WEIGHTS` (must sum to 1.0), `METRICS` dict (per-metric weight, direction, bounds, source), `PILLAR_METRICS` (derived grouping). Weight-sum assertions run at import time.

- **[market_intelligence/src/processing.py](market_intelligence/src/processing.py)** — `load_markets()` validates CSV structure; `normalize_all()` applies direction-aware min-max scaling to produce 0–100 metric scores. Bounds come from config; `lower_is_better` metrics are inverted so a lower raw value always maps to a higher score.

- **[market_intelligence/src/customer_targeting.py](market_intelligence/src/customer_targeting.py)** — Three customer archetypes (`hyperscaler`, `ai_lab`, `enterprise`) each with distinct `pillar_weights`. `score_for_archetype()` temporarily swaps config weights, re-runs scoring, then restores. `best_market_per_archetype()` returns the top market per archetype. `archetype_ranking_comparison()` produces a side-by-side rank table.

- **[market_intelligence/src/scenarios.py](market_intelligence/src/scenarios.py)** — Three alternative scenarios defined as combinations of pillar weight overrides and metric multipliers. `run_scenario()` applies adjustments, re-normalizes, and re-scores. `most_robust_markets()` returns markets that stay in the top half across all scenarios.

- **[market_intelligence/src/risk_assessment.py](market_intelligence/src/risk_assessment.py)** — `RISK_REGISTRY` defines Low/Medium/High ratings per market on regulatory, grid, and execution dimensions. `flag_opportunity_risk_quadrants()` merges risk ratings with the scored DataFrame to produce the 2×2 opportunity/risk classification.

- **[market_intelligence/src/economics.py](market_intelligence/src/economics.py)** — `MARKET_ECONOMICS` stores per-market cost inputs (construction, land, interconnection, power price, PUE, colocation rate). `compute_project_economics()` produces capex, opex, revenue, NOI, and simple payback for a configurable MW capacity. Not a financial model — a structured screening estimate.

- **[market_intelligence/src/visualization_extended.py](market_intelligence/src/visualization_extended.py)** — Three Wave 6 figures: archetype ranking grouped bars, scenario robustness slope chart, opportunity vs. risk 2×2 scatter.

- **[market_intelligence/src/scoring.py](market_intelligence/src/scoring.py)** — `score_markets()` orchestrates: metric scores → pillar scores (weighted avg within pillar) → composite `opportunity_score` (weighted avg across pillars) → rank. `build_scored_table()` assembles the full output with all three tiers in one CSV.

- **[market_intelligence/src/visualization.py](market_intelligence/src/visualization.py)** — Three figures: stacked bar ranking chart (pillar contributions), pillar heatmap (markets × pillars), radar chart (top-N profile comparison). `generate_all()` saves everything at once.

### Scoring design

Four pillars: `demand` 30%, `energy` 30%, `feasibility` 25%, `strategic` 15%.

13 metrics total. Composite score = weighted average of pillar scores; each pillar score = weighted average of its metrics. Pillar contribution to composite = `pillar_score × pillar_weight` — these contributions sum to `opportunity_score` and are used as stacked bar segments in Figure 1.

### Adding a new market

Add one row to `market_intelligence/data/raw/market_inputs.csv` with values for all 13 metric columns, then re-run `run_market_score.py`. Normalization bounds are fixed in config, so adding a new market does not rescale existing markets.
