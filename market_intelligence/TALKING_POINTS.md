# Interview Talking Points — Market Opportunity Engine

These are prepared talking points for interviews with Intersect Power or similar
firms in clean energy development, AI infrastructure, or energy-focused BD roles.
They are written to be accurate, not exaggerated, and to connect the project
work to the actual responsibilities of an early-career BD or strategy analyst.

---

## Talking Point 1 — On the decision problem this project addresses

*Use when asked: "Tell me about a project you've worked on" or "Why did you build this?"*

---

"One of the things I tried to work through analytically is how a developer
like Intersect actually decides which markets to enter before there's a
specific site or customer in hand. The interconnection queue in CAISO is
nearly four years. Land in Northern Virginia is essentially gone. But those
are the markets that get the most attention because they already have
infrastructure. I wanted to build a framework that systematically evaluated
markets on the dimensions that matter to an energy developer specifically —
not just where demand exists today, but where you can actually build and
interconnect, and where your clean energy mandate is reinforced rather than
constrained by the local grid.

The output is a ranking, but the more useful output is the decomposition —
understanding whether a market is attractive because demand is strong, or
because the energy economics work, or both. Those are very different
investment theses."

**Why this works:** It shows you understand that a BD role at an energy developer
is not the same as a BD role at a data center operator or a tech company. The
key constraint is not just demand — it is the intersection of demand, power
supply, and buildability.

---

## Talking Point 2 — On the methodology and its limitations

*Use when asked: "Walk me through how you built this" or "What are the limitations?"*

---

"The framework uses a weighted scoring approach across four pillars: demand
potential, energy attractiveness, deployment feasibility, and strategic fit.
Each pillar has between three and four metrics, and all of them are normalized
to the same 0–100 scale so they're comparable.

The most important design decision was fixing the normalization bounds in the
configuration rather than deriving them dynamically from the dataset. If I
used dataset-relative bounds, adding one new market at the edge of the
distribution would rescale every existing market's score. Fixed bounds, based
on reasonable real-world ranges, keep the scores stable across dataset iterations.

On limitations — the data is public and approximate. Interconnection timelines
are median estimates from LBNL's queue tracker, not project-specific study
outcomes. Wholesale prices are annual day-ahead averages, not the delivered
cost a developer would actually underwrite. And several metrics are ordinal
scores where I made a judgment call from public sources.

What I'd do with more time or access is replace the ordinal scores with
primary research — conversations with developers who've been through the
permitting process in each state. The framework is the right structure; the
data quality is the constraint."

**Why this works:** Interviewers at analytical firms — and Intersect is an
analytical firm — are more impressed by someone who can articulate the
limitations of their own work than by someone who oversells it. This answer
shows methodological awareness and tells them what you'd do next.

---

## Talking Point 3 — On what the results actually show

*Use when asked: "What did you find?" or "Which markets look most interesting?"*

---

"The results surface a few things worth discussing. Pacific Northwest and
ERCOT Texas score near the top under the baseline weights, but for very
different reasons. The Pacific Northwest scores well because of the energy
fundamentals — hydro-dominant baseload, the highest clean energy penetration
in the dataset, cheap and reliable power. Texas scores well because of
feasibility and strategic fit — fastest interconnection queue in the set,
the most active corporate PPA market in the country, and the largest
developable land pipeline for co-located generation.

The more interesting finding for a developer like Intersect is probably MISO
Iowa. It has the best energy economics — average wholesale prices around
$27 per megawatt-hour, the highest wind capacity factors in CONUS — but it
scores low on demand because the existing tech employer base is thin. That
is exactly the kind of market where a developer who can bring an anchor
tenant or a long-term offtake agreement has a structural advantage over
a developer who needs demand to already be there.

Northern Virginia is the clearest example of why the framework is useful.
It has the highest demand signal in the dataset — it is the largest data
center market in the world. But the 42-month interconnection queue and
the near-zero renewable co-location potential score it below every other
market on feasibility. The demand is real, but the development opportunity
for a greenfield clean energy project is not."

**Why this works:** This answer shows you can read your own output critically
and translate it into a business argument. Iowa as the most interesting
contrarian pick is a defensible and specific claim that an interviewer can
engage with.

---

## Talking Point 4 — On the connection to Intersect Power's business specifically

*Use when asked: "Why do you want to work at Intersect?" or "How does your
background relate to what we do?"*

---

"Intersect's model is differentiated from a merchant generator or a pure-play
data center developer because the value proposition is the integration of
clean energy supply and compute demand at the site level. That means the
market selection question is not just 'where is demand?' — it is 'where can
we develop utility-scale renewables in close proximity to a site that meets
a hyperscaler's requirements for power, water, connectivity, and permitting?'

That is the specific question I tried to build into the strategic fit pillar
of this framework. Renewable co-location potential is the highest-weighted
metric in that pillar because it is the variable that most directly reflects
whether Intersect's model can be expressed in a given market, versus whether
you'd be developing a generic data center that happens to buy green power
from somewhere else.

The result is that markets like ERCOT and Iowa rank higher on strategic fit
than their demand scores alone would suggest, and markets like Northern
Virginia — which has all the demand — rank low on strategic fit because
there is no room to build."

**Why this works:** It shows you've done more than a generic project. You've
thought about the specific economics and differentiation of Intersect's
business and built that logic into the framework. An interviewer will notice
that the strategic fit pillar is not generic — it reflects an understanding
of what makes a co-location developer different from a data center REIT.

---

## Talking Point 5 — On what you learned and what you'd do next

*Use when asked: "What would you improve?" or "What did working on this teach you?"*

---

"The clearest gap is sub-market granularity. I scored markets at the ISO/RTO
level, which is the right unit for power price and grid analysis, but
development decisions are made at the county or transmission zone level.
A market that looks attractive at the ERCOT level has very different
conditions in West Texas — where the wind resource is excellent but
transmission is constrained — versus the Dallas-Fort Worth load zone.

The second gap is the absence of a land siting layer. Knowing that a market
has high renewable co-location potential in aggregate doesn't tell you
whether there is a specific parcel of land within reasonable interconnection
distance of a potential data center site. That requires GIS data and
transmission topology analysis that is beyond what public sources easily
support.

What this taught me practically is how interconnected the development
constraints are. I initially thought about power price and interconnection
queue as independent variables. They're not — markets with cheap power
often have cheap power because transmission is constrained and generators
can't get power out, which means the interconnection queue is actually
worse than the low price would suggest. MISO Iowa has both cheap prices
and a reasonable queue because the load is local; you're not trying to
export wind power to Chicago. Understanding that relationship between
price, transmission, and queue was the most useful thing I got from
building this."

**Why this works:** This answer demonstrates genuine analytical engagement
with the material — you learned something real, not just "I improved my
Python skills." The observation about price/transmission/queue correlation
is a sophisticated and accurate point that an energy professional will
recognize as correct.

---

## General guidance for using these talking points

- These are starting points, not scripts. Adapt them to the specific question
  and conversation context.
- When discussing limitations (Talking Point 2), lead with the limitation
  before being asked — it is always more credible than being asked to name
  a weakness and then responding.
- The most important question to be ready for is: "What would you do with
  this if you were actually working here?" The honest answer is: take the
  framework, replace the ordinal scores with primary research and proprietary
  data, add sub-market granularity, and use it as a living document that gets
  updated as queue conditions and power prices change. That is a real answer
  that demonstrates you understand the difference between a portfolio project
  and a production tool.
- Do not claim the data is real-time or that the scores should be taken at
  face value for an actual investment decision. Interviewers at development
  firms know what the data quality of public sources looks like. The project's
  value is in the framework, not the numbers.
