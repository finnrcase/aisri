# Variable Dictionary

Complete reference for all 13 metrics used in the market opportunity scoring
framework. Each entry defines what the variable measures, why it is included,
and what its analytical limitations are.

---

## Demand Potential Pillar (30% of composite)

*Captures the size and character of the AI/data center market in each
geography. High demand scores indicate markets where paying customers already
exist or where structural conditions create near-term demand.*

| Variable | `hyperscale_presence` |
|---|---|
| **Pillar** | Demand Potential |
| **Metric weight** | 30% of pillar |
| **What it measures** | Ordinal score (0–5) for the breadth and scale of hyperscale cloud and AI operator presence (AWS, Azure, Google Cloud, Meta) in the market |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 0–5 |
| **Why it matters** | Hyperscale operators are the primary customers for AI infrastructure. Their existing presence signals that the market has passed the minimum threshold for power supply, connectivity, and permitting that a large data center requires. Markets with no hyperscale presence have not been validated. |
| **Limitation** | Does not distinguish between one small campus and ten large ones. Does not capture announced-but-not-yet-built capacity. |
| **Source** | CBRE Data Center Trends Report; Cushman & Wakefield Global DC Report |

| Variable | `data_center_capacity_mw` |
|---|---|
| **Pillar** | Demand Potential |
| **Metric weight** | 35% of pillar |
| **What it measures** | Total installed and under-construction data center capacity in the market, in megawatts |
| **Direction** | Higher is better |
| **Proxy type** | Continuous |
| **Unit** | Megawatts (MW) |
| **Why it matters** | The single most direct measure of how large the market actually is. A larger market means more potential tenants, more competitive power pricing, and more infrastructure to build on. Also signals absorption risk — a very large market may be near saturation. |
| **Limitation** | Published estimates vary across sources and lag real-time conditions by 6–12 months. Does not capture vacancy rates or absorption pace. |
| **Source** | JLL Data Center Outlook; DC Byte market reports |

| Variable | `tech_employment_index` |
|---|---|
| **Pillar** | Demand Potential |
| **Metric weight** | 20% of pillar |
| **What it measures** | Technology sector employment relative to the U.S. national average (1.0 = national average) |
| **Direction** | Higher is better |
| **Proxy type** | Continuous / indexed |
| **Unit** | Index (1.0 = national average) |
| **Why it matters** | Tech employment concentration is a leading indicator of data center demand. Companies locate infrastructure close to their engineering workforce. High tech employment also signals local AI compute demand from startups, mid-market companies, and cloud-native businesses that are not hyperscalers but are collectively significant. |
| **Limitation** | Employment concentration lags actual compute demand. A market with remote-first tech companies may have dispersed employment but concentrated infrastructure. |
| **Source** | U.S. Bureau of Labor Statistics QCEW; CompTIA State of the Tech Workforce |

| Variable | `fiber_connectivity_score` |
|---|---|
| **Pillar** | Demand Potential |
| **Metric weight** | 15% of pillar |
| **What it measures** | Ordinal score (1–5) for long-haul and metro fiber density, internet exchange point (IXP) presence, and latency to major U.S. population centers |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | Data center infrastructure without high-quality fiber connectivity has limited commercial value for most use cases. AI inference workloads in particular have latency requirements that constrain siting. A market with poor connectivity requires capital investment in fiber before it becomes commercially viable. |
| **Limitation** | Fiber infrastructure is buildable — this score reflects current conditions, not a permanent constraint. A developer with enough capital can pull fiber. The score captures the as-is development cost, not a ceiling. |
| **Source** | FCC Broadband Data Collection; Telegeography fiber maps; PeeringDB |

---

## Energy Attractiveness Pillar (30% of composite)

*Captures the cost, quality, and cleanliness of electricity supply. For a
clean energy developer, this pillar measures both the economic case (cheap
power reduces operating costs) and the strategic case (clean power supports
corporate tenant sustainability commitments).*

| Variable | `avg_wholesale_power_price_mwh` |
|---|---|
| **Pillar** | Energy Attractiveness |
| **Metric weight** | 35% of pillar |
| **What it measures** | Annual average day-ahead wholesale electricity price in the market's ISO/RTO, in $/MWh |
| **Direction** | Lower is better |
| **Proxy type** | Continuous |
| **Unit** | $/MWh |
| **Why it matters** | Power is typically the largest operating cost for a data center, representing 60–70% of PUE-adjusted operating expense. A $10/MWh difference in power price on a 100 MW load is approximately $8.8M per year in operating cost difference. At scale, the market choice is a significant underwriting variable. |
| **Limitation** | Day-ahead spot prices are not the same as the delivered cost of power for a large industrial customer, which includes transmission charges, ancillary services, capacity charges, and any hedging costs. The all-in cost can be 30–60% above the wholesale price depending on the market. |
| **Source** | EIA Electric Power Monthly; FERC LCIA price reports |

| Variable | `renewable_capacity_factor_pct` |
|---|---|
| **Pillar** | Energy Attractiveness |
| **Metric weight** | 30% of pillar |
| **What it measures** | Weighted average P50 capacity factor (%) for wind and solar resources available in or proximate to the market |
| **Direction** | Higher is better |
| **Proxy type** | Continuous |
| **Unit** | Percent (%) |
| **Why it matters** | Capacity factor directly determines the capital cost per megawatt-hour of energy produced from a renewable project. A wind farm with a 45% capacity factor requires roughly 20% less capital per unit of energy output than one with a 35% capacity factor, at the same installed cost. For a co-location developer, the resource quality determines whether the renewable economics can support a competitive power delivery price to the tenant. |
| **Limitation** | Market-level averages mask significant site-specific variation. A 45% average capacity factor for Iowa wind could reflect a range of 38%–55% across actual sites. Project economics depend on site-level resource assessment, not market averages. |
| **Source** | NREL Annual Technology Baseline (ATB); NREL Wind and Solar resource maps |

| Variable | `clean_energy_penetration_pct` |
|---|---|
| **Pillar** | Energy Attractiveness |
| **Metric weight** | 20% of pillar |
| **What it measures** | Share of total electricity generation from clean sources (wind, solar, hydro, nuclear) in the relevant grid region, as a percentage |
| **Direction** | Higher is better |
| **Proxy type** | Continuous |
| **Unit** | Percent (%) |
| **Why it matters** | Higher grid-level clean energy penetration reduces the carbon intensity of grid-delivered power and simplifies a tenant's corporate sustainability accounting. It also indicates that the grid operator has experience managing variable renewable generation, which is relevant to reliability and curtailment risk. |
| **Limitation** | Grid-average clean energy penetration is not the same as what a specific customer receives. A hyperscale tenant with a 24/7 clean power commitment needs to match clean generation to their load hour by hour, which requires specific contractual instruments regardless of grid-level penetration. |
| **Source** | EIA Form 923; EPA eGRID database |

| Variable | `grid_reliability_score` |
|---|---|
| **Pillar** | Energy Attractiveness |
| **Metric weight** | 15% of pillar |
| **What it measures** | Ordinal score (1–5) for grid reliability based on SAIDI/SAIFI outage frequency and duration metrics and ISO/RTO reliability assessments |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | Data centers require extremely high availability — typically 99.999% uptime (Tier IV) or 99.99% (Tier III). Grid-level reliability is the foundation. A market with poor grid reliability shifts the cost of reliability onto the data center operator (backup generation, UPS systems, redundant feeds), which increases capital costs and is a constraint for some tenants. |
| **Limitation** | SAIDI/SAIFI metrics reflect average conditions. Tail-risk events (extreme weather, equipment failure) that matter most for data center siting are not well captured in average reliability statistics. ERCOT's 2021 event would not be reflected in long-run SAIDI averages. |
| **Source** | NERC Long-Term Reliability Assessment; EIA Electric Power Annual (outage data) |

---

## Deployment Feasibility Pillar (25% of composite)

*Captures the practical difficulty of actually building large energy and data
infrastructure in each market. This pillar is often the factor that separates
a commercially attractive market from one that is actually developable.*

| Variable | `interconnection_queue_months` |
|---|---|
| **Pillar** | Deployment Feasibility |
| **Metric weight** | 35% of pillar |
| **What it measures** | Median estimated time in months to clear the generator interconnection queue in the relevant ISO/RTO |
| **Direction** | Lower is better |
| **Proxy type** | Continuous |
| **Unit** | Months |
| **Why it matters** | Generator interconnection is the single most common cause of large energy project delays in the United States. A developer cannot deliver power until interconnection is complete. At 14 months (ERCOT) vs. 48 months (CAISO), the difference in time-to-first-revenue is three years — which at a 10% cost of capital on $500M of invested capital is a material IRR impact. |
| **Limitation** | Median queue times are aggregate statistics. Individual project timelines depend on queue position, transmission topology near the point of interconnection, and the outcome of cluster or sequential study processes that are not predictable at the market screening stage. Queue times are also changing rapidly as the volume of applications has increased sharply since 2020. |
| **Source** | LBNL Interconnection Queue Report (annual); ISO/RTO interconnection study timelines |

| Variable | `permitting_speed_score` |
|---|---|
| **Pillar** | Deployment Feasibility |
| **Metric weight** | 25% of pillar |
| **What it measures** | Ordinal score (1–5) for the ease and speed of state and local permitting for large energy and data infrastructure projects |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | State and local permitting adds 6–36 months to project timelines depending on the jurisdiction. California's CEQA review is the extreme case; Texas's streamlined process for industrial development is the other end of the spectrum. For a developer with capital at risk during the permitting period, faster permitting is directly equivalent to higher returns. |
| **Limitation** | Permitting timelines are highly project-specific and depend on local opposition, site-specific environmental constraints, and the backlog at specific state agencies. Market-level scores reflect the regulatory environment, not the outcome for a specific project. |
| **Source** | Clean Energy States Alliance; state energy office reports; industry surveys |

| Variable | `land_cost_index` |
|---|---|
| **Pillar** | Deployment Feasibility |
| **Metric weight** | 20% of pillar |
| **What it measures** | Relative land cost for large-acreage industrial or agricultural land, indexed so that lower values indicate cheaper land |
| **Direction** | Lower is better |
| **Proxy type** | Continuous / indexed |
| **Unit** | Index (relative, lower = cheaper) |
| **Why it matters** | AI-coupled infrastructure development requires significant land — a 100 MW data center campus with co-located generation might require 500–1,000 acres. The land cost is a component of total development cost that varies by more than an order of magnitude across this dataset (Kansas vs. Northern Virginia). At high land costs, the economics of co-located generation are impaired because land acquisition competes with infrastructure capital. |
| **Limitation** | This is a market-level index, not a specific site cost. Prime locations within a market (sites with existing power access, fiber, and water) will trade at multiples of the market average. |
| **Source** | USDA Land Values Summary; CoStar industrial land comps |

| Variable | `water_availability_score` |
|---|---|
| **Pillar** | Deployment Feasibility |
| **Metric weight** | 20% of pillar |
| **What it measures** | Ordinal score (1–5) for freshwater availability for cooling based on published water stress indicators |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | Data centers with evaporative cooling towers are significant water consumers — a large hyperscale facility can consume 1–5 million gallons per day. In water-stressed markets, this creates permitting risk, regulatory risk, and community opposition. It can also become a genuine physical constraint during droughts. Air-cooled designs reduce but do not eliminate water consumption. |
| **Limitation** | Water stress is assessed at the state or watershed level. Site-specific water availability depends on proximity to water sources and local water rights, which are not captured in market-level scores. |
| **Source** | USGS Water Resources; WRI Aqueduct Water Risk Atlas |

---

## Strategic Fit Pillar (15% of composite)

*Captures alignment between the market's characteristics and the specific
value proposition of a clean energy developer pursuing AI infrastructure
co-location. This pillar is most differentiated from a generic data center
site selection framework.*

| Variable | `renewable_colocation_potential` |
|---|---|
| **Pillar** | Strategic Fit |
| **Metric weight** | 40% of pillar |
| **What it measures** | Ordinal score (1–5) for the availability of large, developable land parcels for utility-scale wind or solar generation that are adjacent to or proximate to viable AI infrastructure sites |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | This is the variable most directly aligned with the co-location development model. The economic and sustainability case for integrating generation and compute at the campus level depends on the ability to develop renewable generation on or near the data center site. A market where that is not possible (Northern Virginia, dense coastal California) is a market where the developer becomes a conventional data center operator rather than an integrated infrastructure developer. |
| **Limitation** | Market-level co-location potential does not confirm that specific sites with the right combination of wind/solar resource quality, grid access, and data center requirements exist. That requires site-level analysis. |
| **Source** | NREL ReEDS model; NREL SolarAnywhere/WindNavigator; USDA NLCD land use maps |

| Variable | `ppa_market_maturity_score` |
|---|---|
| **Pillar** | Strategic Fit |
| **Metric weight** | 30% of pillar |
| **What it measures** | Ordinal score (1–5) for the depth and liquidity of the corporate Power Purchase Agreement market in the state or region |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | A mature PPA market indicates willing counterparties, established legal and regulatory frameworks for energy contracts, and precedent transactions that reduce execution risk. For a developer selling clean power to AI companies, the depth of the PPA market determines how quickly they can find a counterparty and close a transaction. Markets with thin PPA activity require more time and legal cost to execute the first deal. |
| **Limitation** | PPA market maturity is a lagging indicator — it reflects past deal activity, not future demand. A market with high renewable co-location potential but low current PPA activity (Kansas) may develop rapidly if a major tenant commits. |
| **Source** | LevelTen Energy PPA Market Intelligence Report; BNEF Corporate PPA League Tables |

| Variable | `state_policy_score` |
|---|---|
| **Pillar** | Strategic Fit |
| **Metric weight** | 30% of pillar |
| **What it measures** | Ordinal score (1–5) reflecting the favorability of state energy and economic development policy for clean energy and data infrastructure |
| **Direction** | Higher is better |
| **Proxy type** | Ordinal / qualitative |
| **Unit** | Score 1–5 |
| **Why it matters** | State policy creates or removes the structural conditions for clean energy infrastructure development. A strong Renewable Portfolio Standard creates utility demand for renewable power. Data center tax incentives directly reduce tenant operating costs. State-level support for large industrial development accelerates permitting and reduces opposition. Over a 20-year project horizon, the policy environment matters as much as today's economics. |
| **Limitation** | State policy changes. A score based on current policy conditions may be stale within an election cycle. The direction of policy change (toward more or less support for clean energy) is a relevant input that a point-in-time score does not capture. |
| **Source** | DSIRE (Database of State Incentives for Renewables and Efficiency); NCSL energy policy tracker |
