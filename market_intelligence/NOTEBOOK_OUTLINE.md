# Notebook Outline — Market Opportunity Analysis
# market_intelligence/notebooks/market_opportunity_analysis.ipynb

This document describes the intended structure and narrative arc of the analysis
notebook. Each section header maps to a markdown cell in the notebook, followed
by the code cells that produce that section's output.

The notebook is written to be readable by a non-technical reviewer — a BD
director, an infrastructure investor, or a recruiter — without running any code.
The prose cells carry the argument; the code cells are the audit trail.

---

## Section 1 — Purpose and Context

**What this section establishes:**
The problem being solved and why a systematic framework is needed.

**Prose content:**
- AI infrastructure development (data centers, AI campuses) is capital-intensive
  and long-lead. A developer entering a new market is making a multi-year commitment
  before a single megawatt of load is operational.
- Market selection decisions are often made informally or based on where demand
  already exists. This tool makes the prioritization logic explicit and auditable.
- The framework is designed from the perspective of a clean energy developer
  evaluating where to site AI-coupled infrastructure — not a data center operator
  optimizing for latency or an enterprise IT team choosing a cloud region.
- Four questions structure the analysis: Is there real demand here? Can we source
  clean, affordable power? Can we actually build and interconnect? Does this
  align with our strategic mandate?

**What it is not:**
A financial model. This is a screening and prioritization tool. It produces
a ranked list with a transparent rationale, not an IRR. The appropriate next
step after identifying a top-tier market is a market entry feasibility study.

---

## Section 2 — Framework Design

**What this section establishes:**
How the four pillars were chosen, what they measure, and why the weights are
set as they are.

**Prose content:**
- **Pillar definitions:** Demand Potential (30%), Energy Attractiveness (30%),
  Deployment Feasibility (25%), Strategic Fit (15%). Explain the rationale for
  each weight — demand and energy are weighted equally because no project
  survives without both; feasibility is the execution constraint; strategic fit
  is a tie-breaker with directional rather than precise value.
- **Metric selection:** 13 metrics across the four pillars. Each metric was
  selected because (a) it is observable from public data, (b) it is directly
  relevant to the development decision, and (c) it varies meaningfully across
  the eight markets in this dataset.
- **Direction logic:** Some metrics are "higher is better" (renewable capacity
  factor, permitting speed). Others are "lower is better" (wholesale power price,
  interconnection queue months). Direction is encoded in the configuration and
  handled automatically during normalization.
- **Table:** Show the full scoring framework — metric, pillar, weight, direction,
  data source — as a formatted DataFrame.

---

## Section 3 — Data and Assumptions

**What this section establishes:**
Where the data came from, what approximations were made, and what that means
for how confident we should be in the results.

**Prose content:**
- This dataset was populated from public sources: EIA, NREL, LBNL, CBRE, and
  NERC. Values represent approximate 2023–2024 conditions. They are not
  proprietary or real-time.
- Several metrics are ordinal scores (1–5) rather than continuous variables.
  Ordinal scoring was used where the underlying variable is either (a) hard to
  quantify precisely from public data or (b) meaningfully categorical. The
  scoring criteria for each ordinal variable are documented in config.py.
- The eight markets were selected to span the range of relevant conditions in
  the U.S. power market — from the most developed (PJM Northern Virginia) to
  near-greenfield (SPP Kansas). This is intentional: a ranking tool is only
  useful if the inputs vary.
- **Limitations table:** List each metric with its data source and the
  specific limitation or approximation involved. For example:
  - `interconnection_queue_months` is a median estimate; actual timelines
    are project-specific and change as queues move.
  - `avg_wholesale_power_price_mwh` reflects day-ahead spot prices, not the
    all-in delivered cost a developer would actually pay.
  - Ordinal scores reflect the author's judgment based on available data;
    a professional market entry study would replace these with primary research.

**Code output:** Print the raw `market_inputs.csv` as a formatted table with
units and source columns from config.py alongside each metric's raw value.

---

## Section 4 — Normalization Methodology

**What this section establishes:**
How raw values are converted to 0–100 scores, and why min-max scaling with
fixed bounds is the right approach here.

**Prose content:**
- Min-max normalization maps each metric to [0, 100] where 0 = least attractive
  and 100 = most attractive within the defined range. The formula differs by
  direction.
- **Why fixed bounds rather than dataset-relative bounds?** If bounds were set
  dynamically from the current dataset, adding one new market at the extreme
  of the distribution would rescale every existing market's score. Fixed bounds
  (defined in config.py, based on reasonable real-world ranges) make the scores
  stable and comparable across iterations of the dataset.
- **Missing value policy:** A null metric value receives the worst possible
  score (0). This is conservative by design. A developer who cannot source data
  for a market should treat that market with caution, not indifference.
- Walk through one worked example — show ERCOT Texas's raw `avg_wholesale_power_price_mwh`
  (34 $/MWh) being inverted and scaled to a score of ~69.

**Code output:** Table of normalized metric scores for all markets, with the
raw value and score side by side for each metric. Sorted by market_id.

---

## Section 5 — Results: Market Rankings

**What this section establishes:**
The headline output — which markets rank highest and what drives their ranking.

**Prose content:**
- Brief narrative: which market leads, what combination of factors explains it,
  and which markets score low and why.
- Introduce the key insight from the stacked bar chart: composite score =
  sum of weighted pillar contributions. This decomposition shows not just
  rank but the composition of each market's score.
- Note that markets in the middle of the ranking are often the most
  strategically interesting — they are not already saturated (like Northern
  Virginia) but are not purely greenfield (like Kansas).

**Code output:**
- Ranked summary table (rank, market, composite score, four pillar scores)
- Figure 1: Stacked horizontal bar chart of opportunity scores by pillar contribution

---

## Section 6 — Results: Pillar Analysis

**What this section establishes:**
A deeper look at where markets are strong and where they have gaps.

**Prose content:**
- No market scores uniformly high across all four pillars. Every market has
  a dominant strength and at least one material weakness. This is expected and
  useful: it tells a developer what risk they are accepting in each market.
- Discuss specific examples:
  - MISO Iowa: top-tier energy economics (cheap wind, high capacity factors)
    but weak demand (low tech employment, limited existing infrastructure).
    The right market for a developer who brings their own anchor tenant.
  - PJM Northern Virginia: highest demand signal in the dataset, but
    the worst feasibility scores (42-month interconnection queue, exhausted
    land, no co-location potential). A market where developers compete on
    execution speed and relationships, not site selection.
  - CAISO California: strong policy and PPA market, worst feasibility.
    Illustrates that favorable policy alone is not sufficient.

**Code output:**
- Figure 2: Pillar heatmap
- Figure 3: Radar chart (top 4 markets)
- Per-pillar ranked table showing which market leads on each individual pillar

---

## Section 7 — Sensitivity Analysis

**What this section establishes:**
Whether the ranking is robust to changes in the weight assumptions.

**Prose content:**
- Any weighted scoring framework embeds value judgments in its weights.
  The question is not whether the weights are precisely correct — they cannot be —
  but whether the ranking is materially sensitive to reasonable weight changes.
- Test two alternative scenarios:
  1. **Energy-developer scenario:** Up-weight energy attractiveness to 45%,
     reduce demand to 15%. Relevant for a developer who controls the energy
     supply and needs to attract a tenant, rather than chasing existing demand.
  2. **Demand-first scenario:** Up-weight demand to 45%, reduce strategic to 10%.
     Relevant for an operator whose primary constraint is securing paying customers.
- Key finding: the top 2-3 markets are stable across all three scenarios.
  The ranking is most sensitive in the middle tier (ranks 4–6), where the
  weight assumptions determine whether Iowa or Georgia or Ohio is preferred.
  This is an honest and useful result — it tells you where to do more diligence.

**Code output:**
- Side-by-side table: rank under baseline, energy scenario, demand scenario
- Highlight markets whose rank changes by 2+ positions across scenarios

---

## Section 8 — Business Interpretation and Limitations

**What this section establishes:**
What this analysis can and cannot tell a decision-maker. Required for credibility.

**Prose content:**

**What this analysis supports:**
- A first-pass screening of markets to prioritize for further diligence.
- Structured documentation of the factors that matter and how they vary
  across markets — useful as a briefing document before a market entry meeting.
- A framework that can be updated as conditions change (new queue data,
  updated power prices, new state policy) without rebuilding the logic.

**What this analysis does not replace:**
- A site-specific interconnection feasibility study.
- A financial model with project-level capital costs, revenue assumptions,
  and returns.
- Primary market research — conversations with utilities, landowners,
  and potential off-takers.
- Real-time data. Wholesale power prices, interconnection timelines, and
  permitting conditions change. This dataset reflects approximate 2023–2024
  conditions and should be refreshed before any decision is made.

**Final paragraph:**
The value of a framework like this is not in the precise scores — the scores
depend on weights that involve judgment. The value is in forcing a systematic
comparison across a consistent set of criteria, making the implicit logic of
market selection explicit and reviewable.

---

## Appendix — Data Sources and Metric Definitions

**Code output:** Print the full METRICS dict from config.py as a formatted
table with metric_id, pillar, direction, bounds, description, and source.
This is the complete data dictionary for the project.
