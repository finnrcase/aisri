# Methodology Note

## AISRI Market Opportunity Engine — v1

---

### What this framework does

This is a comparative market-prioritization framework. It takes a defined set
of publicly available proxy variables, normalizes them to a common scale, and
aggregates them into a ranked composite score using a transparent weighting
structure. The output is a relative ranking of markets, not an absolute measure
of market attractiveness.

The appropriate use of this framework is to identify which markets warrant
further diligence — site visits, primary research, interconnection pre-screening,
and financial modeling — and which do not. It is a filter, not a conclusion.

---

### Scoring convention

All scores range from 0 to 100. **Higher scores indicate more attractive
markets.** This convention applies at the metric, pillar, and composite level.

This is the inverse of the AISRI risk-scoring convention used elsewhere in
this project, where higher scores indicate higher sustainability risk. The
distinction is intentional and is maintained consistently throughout the
market opportunity module.

---

### Normalization

Each metric is normalized using min-max scaling within bounds defined in
`src/config.py`:

```
For "higher is better" metrics:
    score = (value − lower_bound) / (upper_bound − lower_bound) × 100

For "lower is better" metrics:
    score = (upper_bound − value) / (upper_bound − lower_bound) × 100
```

Bounds are set explicitly in the configuration file rather than derived
dynamically from the dataset. This design choice ensures that scores are
stable when new markets are added — a new market at the edge of the
distribution does not rescale all existing markets.

Values outside the defined bounds are clipped before scoring. A market that
reports a variable above the upper bound still receives a score of 100 on
that metric; it does not receive a score above 100.

Missing values receive the worst possible score (0) on the affected metric.
This is a conservative assumption: a market for which data cannot be sourced
is treated as having no advantage on that dimension rather than being ignored.

---

### Aggregation

Scoring follows a two-level weighted average:

**Level 1 — Metric to pillar:**
Each metric's score is multiplied by its within-pillar weight. The weighted
scores for all metrics in a pillar are summed to produce the pillar score.
Metric weights within each pillar sum to 1.0.

**Level 2 — Pillar to composite:**
Each pillar score is multiplied by its top-level pillar weight. The four
weighted pillar scores are summed to produce the composite opportunity score.
Pillar weights sum to 1.0.

The composite opportunity score therefore equals:

```
opportunity_score =
    Σ over pillars [ pillar_weight × Σ over metrics [ metric_weight × metric_score ] ]
```

This structure means the composite score is fully decomposable — any market's
score can be traced back to the specific metric values that drove it.

---

### Weighting rationale

Weights are not derived from statistical analysis. They reflect considered
judgment about the relative importance of each pillar and metric for a
clean energy developer evaluating AI infrastructure opportunities. They encode
a specific perspective: that demand and energy supply are equally important
first-order constraints (30% each), that execution risk is substantial but
secondary (25%), and that strategic alignment is a meaningful differentiator
but not the primary filter (15%).

Alternative weight schemes produce different rankings, primarily in the
middle tier (ranks 3–6). The sensitivity analysis in the notebook demonstrates
this directly. The top two markets (Pacific Northwest and ERCOT Texas under
baseline weights) are stable across the three weight scenarios tested.

The weights can and should be adjusted based on the specific investment thesis
being evaluated. A developer who supplies power and needs to find a tenant
should up-weight energy attractiveness. A developer who follows tenant demand
should up-weight demand potential. The framework supports both by making weights
a single parameter in the configuration file.

---

### Data quality

All input data is drawn from public sources and represents approximate
2023–2024 conditions. The dataset has the following limitations that affect
how the outputs should be interpreted:

**Continuous variables** (wholesale power price, capacity factor, capacity MW,
employment index) are sourced from published reports and represent point-in-time
estimates. They are not real-time and should be refreshed before any decision
is made.

**Ordinal variables** (hyperscale presence, fiber connectivity, permitting speed,
grid reliability, co-location potential, PPA market maturity, state policy) are
scored on a 1–5 or 0–5 scale based on the author's assessment of publicly
available evidence. The scoring criteria for each variable are documented in
`src/config.py` and `docs/variable_dictionary.md`. These scores reflect
the current state of the evidence base as of the dataset construction date
and are subject to revision as conditions change.

No proprietary or non-public data was used in constructing this dataset.

---

### What this framework does not do

This framework does not produce:

- A financial model or return estimate for any specific project or market
- A site-specific assessment of interconnection, permitting, or construction feasibility
- A real-time or forward-looking view of market conditions
- An assessment of individual project or counterparty risk

The output is a market-level screening score intended to inform the allocation
of diligence resources, not to replace that diligence.

---

### Extending the framework

The framework is designed to be extended without modifying the scoring logic.
To add a new market, add one row to `data/raw/market_inputs.csv` with values
for all 13 metric columns and re-run `run_market_score.py`.

To add a new metric, define it in `src/config.py` (including pillar assignment,
weight, direction, and bounds), add the column to `market_inputs.csv`, and verify
that all metric weights within the affected pillar still sum to 1.0. The scoring
code does not need to change.

To adjust weights, edit `PILLAR_WEIGHTS` or the `"weight"` field within `METRICS`
in `src/config.py`. The validation function in that file will raise an error
immediately if weights do not sum correctly.
