# Business Interpretation — Market Opportunity Results

This document translates the scored output into plain business language.
Each market is assessed on what the score means in practice, what the
development opportunity actually is, and what tradeoffs a team would be
accepting by pursuing it.

Scores reference the v1 dataset (approximate 2023–2024 conditions).
Interpretation is written from the perspective of a clean energy developer
evaluating AI-coupled infrastructure opportunities.

---

## Rank 1 — Pacific Northwest (WA) | Opportunity Score: ~66

**The core thesis:** The Pacific Northwest offers the most reliable, cleanest,
and cheapest power in the dataset, paired with a proven tech-sector tenant base
and a permitting environment that is materially faster than the coastal markets
it competes with.

**Why it scores well:**
The energy score is the highest in the set. Hydro-dominant generation means
approximately 78% clean energy penetration with baseload characteristics that
wind and solar cannot match — the power is available when you need it, not
just when the resource is producing. Average wholesale prices around $29/MWh
are among the cheapest in the dataset. Grid reliability, measured on SAIDI/SAIFI
metrics, is the best of any market here.

The demand signal is credible. Microsoft and Amazon are headquartered in Seattle.
Google has operated Pacific Northwest infrastructure for years. This is not a
market where a developer has to build speculative capacity and wait for tenants —
the tenants are already there and actively expanding.

**The real tradeoffs:**
Hydro is a long-run climate risk. Lower snowpack projections for the Columbia
River basin mean the baseload advantage could erode over a 20–30 year project
horizon. A developer financing infrastructure today should stress-test revenue
assumptions against reduced hydro generation scenarios.

Land cost is above the median in this dataset. The Pacific Northwest is not a
cheap-land story the way Iowa or Kansas is. Development economics depend on
securing sites at reasonable cost before the market gets more competitive.

The renewable co-location potential is good but not exceptional — wind and solar
resources in Washington are real but not at the scale or capacity factor of the
central plains. A developer relying on adjacent renewable generation to back a
clean power claim would likely need to source PPAs from further afield.

**What a BD team should do:**
This is a market where relationship-building with utilities and site owners is
the near-term priority. Puget Sound Energy and Pacific Power are the relevant
regulated utilities. The interconnection queue through the NWPP is faster than
PJM or CAISO, but that advantage compresses if the market heats up further.
The window to establish a low-cost land position is not unlimited.

---

## Rank 2 — ERCOT Texas | Opportunity Score: ~64

**The core thesis:** Texas offers the best execution environment in the U.S.
for large energy infrastructure — fastest interconnection, most permitting-
friendly regulatory environment, lowest time-to-market — combined with the
deepest corporate PPA market and the largest developable renewable pipeline.
The demand story is real and growing. The headline risk is the grid itself.

**Why it scores well:**
ERCOT's interconnection queue median is roughly 14 months — by far the shortest
in this dataset and well below the national average. Texas permitting scores
maximum on this framework's scale. For a developer whose capital is deployed
at first power delivery, that time advantage compounds into a materially better
IRR than any other market.

The strategic fit score reflects Texas's position as the most active corporate
PPA market in the country. Tech companies, industrials, and financial institutions
have all executed large-scale renewable PPAs in ERCOT. The transaction precedent
is deep, counterparties are sophisticated, and deal structures are well understood.

The renewable co-location potential is the highest in the dataset alongside Iowa
and Kansas. West Texas wind and utility-scale solar in the Permian and South
Texas regions offer large contiguous land parcels, high capacity factors, and
established development ecosystems. The infrastructure to move power from
resource zones to load centers is being built.

**The real tradeoffs:**
The February 2021 winter storm is the correct starting point for any grid
reliability discussion in Texas. ERCOT's isolated interconnection means there
is no emergency import capacity from neighboring grids when generation fails
under stress. Weatherization reforms have been implemented, but the
structural isolation of ERCOT remains. A hyperscale tenant with 24/7 clean
power requirements will ask about backup power assumptions in detail.

Water is a genuine constraint in parts of Texas. Dallas-Fort Worth is less
stressed than West Texas, but as data center concentration increases and
climate-driven droughts persist, water availability for evaporative cooling
will require site-specific diligence. Air-cooled designs mitigate but do not
eliminate this.

The wholesale power market is volatile. ERCOT's energy-only market structure
means there is no capacity payment to generators — revenues are entirely
dependent on spot prices, which historically spike during high-demand periods.
A developer should not underwrite long-run average prices without accounting
for the distribution of outcomes.

**What a BD team should do:**
This is a market where Intersect's model of integrating generation and compute
is most directly applicable. The land, the resource, and the transaction
infrastructure are all present. The priority is identifying sites where
transmission capacity is available or acquirable, water is manageable, and
the proximity to load centers supports the reliability requirements a
hyperscale tenant would impose. West Texas and the Dallas-Fort Worth corridor
offer different risk-return profiles and should be evaluated separately.

---

## Rank 3 — MISO Iowa | Opportunity Score: ~60

**The core thesis:** Iowa has the best energy economics in the dataset —
the combination of wind capacity factors and wholesale prices is unmatched —
but the demand signal is weak by the standards of the established data center
markets. This is a developer's market, not a tenant's market. The right
strategy requires bringing a customer thesis to the market, not following
one that already exists.

**Why it scores well:**
Iowa wind capacity factors approach 45% — a P50 estimate that significantly
exceeds what is achievable in most of the eastern and coastal markets.
Average wholesale prices around $27/MWh reflect both the low marginal cost
of wind generation and the structure of MISO's market, which has historically
been one of the more liquid and competitive wholesale markets in the country.

Clean energy penetration is approximately 62%, the highest among non-hydro
markets in this dataset. An AI developer or corporate tenant with a 24/7
clean power requirement can source a high percentage of their energy from
wind resources that are geographically proximate rather than relying on
RECs or synthetic matching instruments.

The renewable co-location potential reflects what Iowa actually is: large,
flat, agricultural land with consistent wind exposure, cooperative landowners
with experience in wind development, and a state-level policy environment
that has historically supported energy infrastructure.

**The real tradeoffs:**
The demand score is the central weakness. Iowa does not have a deep
tech-sector talent base or a concentrated hyperscale presence. Google
has operated data centers in Iowa since 2007, which validates the
infrastructure case, but a second or third tenant would likely need to
be recruited actively rather than attracted by existing market momentum.

Fiber connectivity outside of the Des Moines metro area is limited compared
to the established data center markets. This is a solvable problem — fiber
can be pulled — but it is a capital cost that reduces the economic advantage
of cheap power.

**What a BD team should do:**
Iowa is a strong candidate for a developer-led model where Intersect secures
a site, develops the generation, and brings a pre-identified tenant rather
than entering a competitive existing market. The economics of cheap wind
power and low land cost support an attractive power delivery price that a
hyperscale tenant should find competitive with markets where they are already
paying $42–58/MWh for dirtier power. The pitch is the delivered economics,
not the ecosystem.

---

## Rank 4 — Southeast Georgia | Opportunity Score: ~53

**The core thesis:** Georgia is an institutional-quality regulated utility
market with improving demand fundamentals and a business-friendly development
environment. It is not a top-tier energy market by resource quality, but
the reliability of regulated utility service and the growth of the Atlanta
tech corridor make it a credible near-term opportunity, particularly for
a developer who values regulatory predictability.

**Why it scores where it does:**
Georgia Power (Southern Company) provides regulated utility service with
reliability metrics among the best in the Southeast. The Vogtle nuclear
expansion adds firm clean power to the state's generation mix — relevant
for tenants with clean energy requirements. Permitting is faster than the
northern PJM markets and California.

The demand signal has improved recently. Microsoft and Google have both
announced significant data center investments in metropolitan Atlanta.
That announced capacity creates the kind of local vendor and workforce
infrastructure that typically accelerates subsequent development.

**The real tradeoffs:**
Clean energy penetration overall is low — approximately 18%, reflecting
a generation mix that is still predominantly natural gas and coal.
A developer claiming clean power delivery would need to be specific about
the sourcing mechanism, since grid-average power in Georgia does not
support a clean power claim without PPAs or on-site generation.

Renewable resource quality is the lowest in this dataset. Solar capacity
factors in Georgia are real but not exceptional. The state has no
meaningful wind resource.

---

## Rank 5–6 — PJM Ohio and SPP Kansas | Opportunity Score: ~51–52

**PJM Ohio** is a market in transition. The AWS Columbus and Meta New Albany
campuses have established a data center presence, but the market is not
yet at the scale where it creates its own momentum. Ohio's RPS is among
the weakest in the country, which limits the policy tailwind. The Great
Lakes water availability is a genuine advantage as water stress becomes
a more significant constraint elsewhere. This is a watchlist market — worth
tracking as PJM's interconnection backlog clears and Ohio's generation mix
evolves, but not yet a clear priority.

**SPP Kansas** presents a paradox: it has the best renewable resource quality
in the dataset — capacity factors approaching 48% for wind — the cheapest
land, and a fast interconnection queue, but essentially no existing data
center market. It scores low on demand for the same reason that it is
attractive on resource: it is underpenetrated. This is a five-to-seven-year
opportunity, not a two-to-three-year one. The right move is to understand
what it would take to attract an anchor tenant before making a site
commitment.

---

## Rank 7–8 — PJM Northern Virginia and CAISO California

These two markets represent opposite failure modes for a greenfield developer.

**PJM Northern Virginia** has more existing data center demand than any other
market in the world — approximately 3,500 MW of installed capacity in the
Ashburn / Prince William County corridor. But the supply side has essentially
closed. Land for large-format development is scarce and expensive. The PJM
interconnection queue is the longest in this dataset (42-month median).
Renewable co-location potential within Northern Virginia is near zero —
there is no land to develop adjacent generation, which means a developer
claiming clean power delivery is dependent on PPAs from distant resources.
This market rewards operational incumbents. It is not a greenfield opportunity.

**CAISO California** is the inverse problem. The policy environment is the
strongest in the country. The PPA market is deep. Demand from tech companies
is real. But the California Environmental Quality Act (CEQA) review process,
the most constrained interconnection queue in the dataset at 48 months,
scarce and expensive land, and chronic water stress make new greenfield
development economically very difficult. California remains the most
important state for corporate clean energy commitments and PPA execution —
but the infrastructure to serve that demand is increasingly being built
outside the state.

---

## Cross-cutting observations for a BD audience

**The interconnection queue is the binding constraint in more markets than
power price or land cost.** PJM and CAISO queue times of 30–48 months are
not primarily a function of technical complexity — they reflect a structural
mismatch between the volume of interconnection applications and the capacity
of the utilities to process studies. A developer who can manage the queue
process efficiently — through early filing, co-development agreements, or
behind-the-meter configurations — has a structural advantage over developers
who treat interconnection as a downstream problem.

**Markets that rank high on energy economics tend to rank low on demand, and
vice versa.** This is not a coincidence. Cheap power markets (Iowa, Kansas,
Pacific Northwest) have cheap power in part because large industrial load
has not yet competed for it. Markets with concentrated tech demand (Northern
Virginia, California) have attracted enough load to push up prices and exhaust
the easy development inventory. The developer opportunity is in finding markets
that are early enough on the demand curve that the energy economics are still
favorable — ERCOT and the Pacific Northwest are the clearest examples of
markets that have real demand without yet being overbuilt.

**Water is an underweighted risk in most market assessments.** This framework
scores water availability as one input among four in the feasibility pillar.
In a 10-year view, it may deserve heavier weighting. The data centers being
permitted today will be operating in 2035, when drought conditions in Texas,
the Southwest, and parts of the Southeast are projected to be materially worse
than historical averages. Air-cooled designs and closed-loop cooling systems
reduce but do not eliminate this exposure.
