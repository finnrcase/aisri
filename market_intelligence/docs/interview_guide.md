# Interview Guide — Market Opportunity Engine

Scripts and Q&A for presenting this project in a BD or strategy interview,
specifically in the context of a clean energy / AI infrastructure role.

---

## 60-Second Explanation

*Use this when asked "tell me about a project you've built" in a screening call
or when you have limited time. Practice until it sounds natural, not recited.*

---

"I built a market prioritization framework for AI-coupled energy infrastructure —
essentially a tool to answer the question of where a clean energy developer should
focus resources when evaluating new markets for data center co-location projects.

The framework scores eight U.S. power markets across four dimensions: demand
potential, energy attractiveness, deployment feasibility, and strategic fit.
Each dimension has three to four metrics drawn from public sources — EIA for
power prices, NREL for renewable resource data, LBNL for interconnection queue
times, and industry reports for the demand-side variables.

The output is a ranked scorecard with a full breakdown by pillar so you can see
not just which market ranks highest but what's driving that ranking. The Pacific
Northwest and ERCOT Texas come out on top under the baseline weights, for
different reasons — the Pacific Northwest wins on energy and reliability, Texas
wins on execution speed and strategic fit.

The more useful output is the market classification: which markets are actionable
now, which need monitoring, and which have strong fundamentals but a specific
constraint that makes them a medium-term story rather than a near-term one."

---

## 2-Minute Interview Walkthrough

*Use this when an interviewer asks you to walk them through the project with
your screen shared or a printout in front of them. The sequence matters —
start with the business problem, show the framework, then the results.*

---

**[0:00–0:20] Open with the business problem, not the tool**

"The starting point for this project was a specific decision problem that a
developer like Intersect faces regularly: you have a limited amount of BD
bandwidth, and there are fifteen U.S. markets you could theoretically pursue.
How do you decide which two or three to focus on first?

The typical answer is informal — you follow existing relationships, or you go
where the demand already is, or you build where you've built before. I wanted
to make that logic explicit."

**[0:20–0:45] Explain the framework structure**

*[Show the heatmap or the scoring table]*

"The framework has four pillars. Demand potential and energy attractiveness are
each weighted at 30% because I think they're co-equal constraints — a project
without demand doesn't pencil regardless of the energy economics, and a project
with demand but in a market where power is expensive or unavailable doesn't work
either. Deployment feasibility is 25% — it's real and often overlooked — and
strategic fit is 15%.

The strategic fit pillar is the one that's most specific to Intersect's model.
The highest-weighted metric in that pillar is renewable co-location potential —
whether there's actually land nearby to develop generation adjacent to the
compute campus. That's the variable that most directly reflects whether the
integrated model is executable in a given market."

**[0:45–1:20] Walk through the key results**

*[Show the ranked bar chart]*

"The Pacific Northwest comes out first. It has the highest clean energy
penetration in the dataset — mostly hydro — the best grid reliability, and
wholesale prices around $29 per megawatt-hour. Microsoft and Amazon are both
headquartered there. The main risk I'd flag is the long-run hydro outlook as
snowpack declines.

Texas is second, and the reason is different. The energy economics are good but
not exceptional. Texas wins on execution — the interconnection queue is 14
months, compared to 42 months in PJM or 48 months in CAISO. At a 10% cost of
capital, three years of additional development time on a $500 million project
is a material IRR difference.

The interesting tension is Iowa. The energy economics are actually the best in
the dataset — $27 per megawatt-hour average wholesale price, 45% wind capacity
factors. It ranks third overall but only because the demand side is thin. That
tells you it's a market where the development opportunity depends on bringing
a tenant rather than following one."

**[1:20–1:45] Show the sensitivity analysis**

*[Show the sensitivity table from the notebook]*

"One thing I tested was how sensitive the ranking is to the weight assumptions.
If you up-weight energy attractiveness to 45% — which is the right lens for
a developer who controls the energy supply and is trying to attract a tenant —
the top two markets stay the same but Iowa moves up and Kansas becomes more
competitive.

The top two markets are stable across every weight scheme I tested. The middle
tier shuffles around. I think that's an honest result — it tells you where to
be confident and where you'd need more market-specific diligence before
committing."

**[1:45–2:00] Close with limitations and next steps**

"The main limitation is data quality. Several variables are ordinal scores I
derived from public sources, not site-specific primary research. The right follow-on
to this kind of screening tool is a more detailed feasibility analysis for the
Priority markets — site identification, preliminary interconnection pre-screening,
and utility conversations. This tells you where to have those conversations first."

---

## Likely Follow-Up Questions and Strong Answers

---

**Q1: "Why did you choose these four pillars? Why not include construction cost,
or proximity to population centers, or latency requirements?"**

"The four pillars reflect a developer's perspective specifically, as opposed to
an operator's or a tenant's. A data center operator optimizing for latency would
weight proximity to population centers heavily. A tenant's procurement team
would weight redundancy and tier certification. I was trying to simulate how a
developer prioritizes markets before they know which specific tenant they're
building for — so I focused on the structural conditions that make a market
developable and commercially viable broadly, rather than optimizing for a
single tenant's requirements.

Construction costs are real, but they're primarily driven by local labor markets
and materials logistics, and they vary less across U.S. markets than power prices
or interconnection timelines. I'd add them in a more detailed version, but at
the screening stage they're a second-order consideration.

Latency is a valid dimension for some workloads. AI training workloads are not
latency-sensitive in the same way that financial trading or consumer applications
are — you don't need to be in New York to train a large language model. If I
were screening for inference-focused infrastructure, I'd add a latency-to-load
metric. For the training and large-scale compute use case, I left it out
intentionally."

---

**Q2: "The interconnection queue estimates you're using are medians. Isn't the
actual timeline highly variable and project-specific?"**

"Yes, that's correct, and it's an important limitation. Median queue times from
LBNL's annual tracker are a reasonable first-order signal — they reflect the
overall backlog in the ISO/RTO and the typical pace at which that backlog clears.
But individual project outcomes depend on queue position, the specific transmission
topology at the point of interconnection, and whether the project enters a cluster
or sequential study process. A project at the front of the CAISO queue could
clear in 18 months; one at the back could wait seven years.

The right way to upgrade this metric for a real development assessment is to
run a preliminary interconnection feasibility study — not a full application,
but a pre-application analysis with the ISO/RTO — for specific substations in
each market. That would give project-specific timeline estimates rather than
market-level averages. That's the first thing I'd replace with primary research."

---

**Q3: "You ranked Texas second. What's your bear case for Texas?"**

"Two things. The first is grid reliability. The 2021 winter storm is the
right frame — ERCOT's isolated interconnection means there's no emergency
import capacity from neighboring grids when generation fails under stress.
Weatherization reforms have been implemented and ERCOT has added significant
backup capacity since 2021, but the structural isolation remains. A hyperscale
tenant with 99.999% uptime requirements is going to ask detailed questions
about backup power configuration, and the answers in Texas require more
redundant on-site generation than they would in a market with cross-border
interconnections.

The second is water. Western Texas has real water stress, and the data center
concentration in the Dallas-Fort Worth area is adding significant water demand
to a region that already faces drought risk. Air-cooled designs help, but the
largest AI training facilities are still water-intensive. Over a 15-20 year
project horizon, water availability in Texas is a more significant risk than
it looks today."

---

**Q4: "What would you do differently if you were building this with Intersect's
internal data rather than public sources?"**

"Three things specifically.

First, I'd replace the ordinal scores with primary research. Permitting speed,
co-location potential, and PPA market maturity are all ordinal proxies in this
dataset because I couldn't get precise continuous data from public sources.
With Intersect's development experience and relationships, you'd replace those
with quantitative estimates based on actual project experience in each market —
months to permit approval, specific site costs, actual PPA bid depth from
LevelTen or similar.

Second, I'd add sub-market resolution. ERCOT is not one market — West Texas,
the Dallas-Fort Worth load zone, and Houston have different power prices,
transmission constraints, and development conditions. Market-level scoring
misses that. A production version of this tool would score at the level of
transmission pricing zones or counties, not ISOs.

Third, I'd add a financial overlay. The scoring tool tells you which markets
are structurally attractive; a financial model would tell you which are
economically attractive at current construction costs, land prices, and
expected revenue. Connecting the two would let you identify markets where the
structural score and the economics disagree — which is often where the
interesting development opportunities are."

---

**Q5: "This is a weighted scoring model. Couldn't you get almost any result
you want by adjusting the weights?"**

"In principle, yes — any weighted scoring model is sensitive to the weights,
and you could construct weights that produce any desired ranking. That's a
legitimate critique of the methodology.

The honest response has two parts. First, I tested three weight schemes
explicitly — the baseline, an energy-developer scenario with energy at 45%,
and a demand-first scenario with demand at 45%. The top two markets are the
same in all three scenarios. The ranking is sensitive in the middle tier but
not at the top. If the conclusion of the analysis is 'pursue Texas and the
Pacific Northwest first,' that conclusion holds across all three weight schemes.

Second, the value of a framework like this isn't the scores — it's that the
framework forces you to be explicit about what you're optimizing for. If
someone disagrees with the result, they have to engage with the specific
weights and variable choices rather than just asserting a different intuition.
That's a more productive conversation than debating rankings without an
explicit framework. In a BD context, the framework is the artifact that
creates alignment on criteria before the decision is made."

---

## Notes on delivery

**On being asked about data quality:**
Lead with the limitation, don't wait to be asked. Saying "the interconnection
data is a median estimate, not a project-specific number" before the interviewer
points it out reads as analytical honesty. Waiting to be challenged reads as
not having thought it through.

**On the sensitivity analysis:**
This is the strongest analytical point in the project. If the interview has time
for only one quantitative discussion, make it the sensitivity analysis. Showing
that you tested your own conclusions is a more sophisticated analytical behavior
than presenting results as if they were certain.

**On Intersect specifically:**
The renewable co-location potential metric is the most Intersect-specific
variable in the framework. Naming it, explaining what it captures, and connecting
it to Intersect's integrated development model demonstrates that you understand
the company's differentiated value proposition. Every other variable in this
framework could appear in a generic data center site selection model. That one
could not.

**On the project's scope:**
Be straightforward about what this is: a portfolio project built with public
data that simulates the kind of analysis a BD analyst would run. It is not a
production tool and the data is not real-time. An interviewer at a development
firm knows what public data quality looks like. The project demonstrates
analytical thinking, domain knowledge, and familiarity with the relevant
variables — not access to proprietary information.
