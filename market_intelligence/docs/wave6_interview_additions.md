# Wave 6 — Interview Additions

How to explain the four new analytical layers to a BD or strategy interviewer.

---

## How to frame the additions as a whole

"After building the baseline market scoring framework, I added four analytical
layers that I think better reflect how a BD team at a development company would
actually use this kind of tool. The core scoring model answers 'which markets
are attractive.' The extensions answer: 'attractive for which customer?',
'how sensitive is that conclusion to conditions changing?', 'what are the
execution risks we'd be accepting?', and 'do the economics roughly work?'
Those four questions are the ones that would come up in an internal review
meeting before committing resources to market entry."

---

## Customer Targeting Layer

**What it does:** Re-runs the pillar scoring with archetype-specific weights
for three customer types — hyperscalers, frontier AI labs, and enterprise
compute clusters — producing a separate ranking per customer type.

**Key insight to communicate:** The same market can rank very differently
depending on who you're trying to attract. Iowa ranks third on the baseline
but jumps to first or second for an AI lab whose primary decision variable
is power cost. Northern Virginia, which ranks seventh on the baseline due
to feasibility constraints, becomes more competitive for a hyperscaler that
has already solved its power and land procurement and just needs to add
capacity to an existing campus.

**Interview answer for "how did you choose the archetype weights":**
"The weights reflect the different cost structures of each customer type.
For an AI lab running large training runs, power is the dominant operating
expense — a 100 MW cluster running for a year consumes over a million
megawatt-hours. At $27/MWh vs. $58/MWh, that's a $31M annual cost
difference that dwarfs most other location factors. So I weighted energy
at 50% for that archetype. For a hyperscaler, they have dedicated
procurement teams and existing clean energy portfolios — energy is a
real cost but not their primary site selection filter. They care more
about whether the market ecosystem exists and whether they can execute
at scale, so demand gets 40%."

---

## Scenario Analysis

**What it does:** Re-runs the full scoring pipeline under three alternative
structural assumptions and identifies which markets are stable across all
scenarios ("robust") vs. which are sensitive to specific assumptions.

**Three scenarios:**
1. High AI Demand Growth — demand pillar gets up-weighted; data center
   capacity inputs scale up 40%
2. Rising Power Prices (+35%) — energy pillar gets up-weighted; all power
   prices increase uniformly
3. Tighter Grid Constraints — feasibility gets up-weighted; interconnection
   queue times lengthen 50%

**Key insight to communicate:** "Pacific Northwest and ERCOT Texas are in
the top three under every scenario I tested. That gives you more confidence
that those are the right markets to prioritize — the conclusion doesn't
depend on any one structural assumption being correct. Iowa gets interesting
under the rising power price scenario, which makes sense because cheap
hydro and wind power become more valuable relative to expensive CAISO
and PJM markets when the absolute price difference widens."

**Interview answer for "aren't your scenarios arbitrary?":**
"The specific multipliers are illustrative — I'm not claiming that power
prices will increase exactly 35%. What the scenario analysis tests is
directional sensitivity: if conditions move in a specific direction, do
the market rankings change materially? For the top-ranked markets, they
don't. For markets in the middle tier, they do — and that's useful
information, because it tells you those markets require a specific
thesis about how conditions will evolve rather than being broadly
attractive."

---

## Risk Assessment

**What it does:** Rates each market on three dimensions — regulatory risk,
grid risk, execution risk — with Low/Medium/High ratings, specific
rationale, and a "watch" condition that would change the rating.

**Key insight to communicate:** "Risk and opportunity are separate
questions. ERCOT Texas scores near the top on opportunity but High on
grid risk. The Pacific Northwest scores near the top on opportunity and
Low across all three risk dimensions. That distinction matters for
capital allocation — a developer should be more willing to commit to
the Pacific Northwest first precisely because the risk profile is
cleaner, even if Texas has comparable opportunity."

**Interview answer for "how did you assess the risks?":**
"The risk ratings are based on the same public sources as the scoring
inputs, but interpreted differently. Grid reliability metrics from NERC
tell you about average conditions; the 2021 ERCOT event tells you about
tail risk. Permitting speed scores from the scoring framework feed into
execution risk, but I also incorporated factors that don't appear in the
scoring — like the Loudoun County zoning debate for Northern Virginia,
which is a specific execution risk that isn't captured in any aggregate
metric. The ratings are my interpretation of the evidence base, not
algorithmic outputs."

**The opportunity/risk quadrant output:**
The 2×2 scatter (opportunity score on x-axis, risk level on y-axis)
is the single most useful chart for a 5-minute executive briefing.
The bottom-right quadrant (High Opportunity / Low Risk) is where you
want to spend development capital. The top-right (High Opportunity /
High Risk) is where you go if you have a specific capability advantage —
like ERCOT, where a developer with strong grid reliability solutions
and water management expertise can capture the opportunity while managing
risks that are prohibitive for less sophisticated operators.

---

## Project Economics

**What it does:** Computes a first-order cost-revenue-payback estimate
for a 100 MW IT load development in each market. Covers total capex,
annual power cost, annual revenue, net operating income, and a simple
(undiscounted) payback estimate.

**Important framing:** This is NOT a financial model. Say that directly
in an interview before they ask. "I want to be clear that this isn't a
DCF — there's no financing structure, no tax assumptions, no depreciation,
and the revenue figures are based on market-rate colocation pricing that
would need to be validated against actual lease negotiations. What this
does show is which markets have structurally better economics and what
the primary driver of the cost differences is."

**Key findings to communicate:**
- Power cost is the largest single operating expense in every market —
  it exceeds O&M by roughly 5:1 at typical colocation pricing. This
  validates why energy attractiveness is weighted so heavily.
- California's high construction costs and power costs make its payback
  the longest in the dataset despite having the highest revenue rates.
  This is the quantitative version of the qualitative argument that
  California demand doesn't translate to a greenfield development opportunity.
- Iowa and Kansas have the shortest payback estimates but the most
  uncertain revenue assumptions — a developer would need a committed
  tenant before underwriting those revenue figures.

**Interview answer for "why not build a full financial model?":**
"A full financial model requires assumptions about capital structure,
financing terms, depreciation schedules, and tax treatment that I don't
have a basis to make without actual project data. Adding those assumptions
wouldn't add analytical value — it would add apparent precision without
real precision. What this economics layer does is identify which
variables drive the cost differences across markets, which is the useful
first-order insight. A real project model would be built with actual
construction bids, utility rate schedules, and a signed term sheet."

---

## How the additions work together

The four layers are designed to answer a sequence of questions that build
toward an actionable recommendation:

1. **Opportunity score** → Which markets are structurally attractive?
2. **Customer targeting** → Attractive for whom, specifically?
3. **Scenario analysis** → Does that conclusion hold if conditions change?
4. **Risk assessment** → What are we accepting if we proceed?
5. **Economics** → Do the numbers roughly support the thesis?

A market that scores well on all five is a Priority. A market that scores
well on opportunity and customer fit but poorly on risk and economics needs
more specific diligence. That structure is how a real BD team would think
about market entry — it's not just "what's the score?" but "what's the full
picture before we commit resources?"
