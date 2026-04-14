# market_intelligence/src/risk_assessment.py
#
# Risk Assessment Layer
#
# The opportunity score measures how attractive a market is. This module
# measures how risky it is to develop there. These are separate questions.
# A market can be attractive AND risky (ERCOT Texas), or unattractive AND
# low-risk (PJM Ohio is unlikely to surprise you in either direction).
#
# Three risk dimensions:
#   Regulatory risk  — permitting changes, state policy reversals, utility decisions
#   Grid risk        — interconnection delays, reliability events, curtailment
#   Execution risk   — land acquisition, construction, water permits, local opposition
#
# Risk ratings are: Low / Medium / High
# Each rating has a short rationale and a specific risk event to watch.
#
# Risk is NOT incorporated into the opportunity score — combining opportunity
# and risk into one number destroys information. Instead, risk is used to
# generate a "high opportunity, high risk" flag that a BD team should treat
# as requiring additional diligence before committing capital.

from __future__ import annotations

import pandas as pd


# ── Risk registry ─────────────────────────────────────────────────────────────
#
# Structure per market:
#   {dimension}: {
#       "rating":    "Low" | "Medium" | "High"
#       "rationale": one sentence explanation
#       "watch":     the specific event or condition that would change this rating
#   }

RISK_REGISTRY: dict[str, dict] = {

    "ercot_tx": {
        "regulatory": {
            "rating":    "Low",
            "rationale": "Texas has no state RPS, a stable pro-development regulatory "
                         "environment, and no CEQA-equivalent review process. Legislative "
                         "risk for energy infrastructure is lower than any other market "
                         "in this dataset.",
            "watch":     "Any federal or state legislative action restricting data center "
                         "water use or imposing a state RPS with onerous compliance timelines.",
        },
        "grid": {
            "rating":    "High",
            "rationale": "ERCOT's isolated interconnection has no emergency import capacity "
                         "from neighboring grids. The 2021 winter storm demonstrated the "
                         "tail risk of weather-driven generation failure. Weatherization "
                         "improvements have been implemented but the structural isolation "
                         "remains. An energy-only market structure also creates revenue "
                         "volatility for generation assets.",
            "watch":     "Extreme weather events that stress the grid beyond weatherized "
                         "capacity limits, or ERCOT reserve margin falling below NERC "
                         "planning standards.",
        },
        "execution": {
            "rating":    "Low",
            "rationale": "Fastest interconnection queue and most permitting-friendly "
                         "environment in the dataset. Texas has deep experience with "
                         "large-format energy infrastructure construction.",
            "watch":     "Water permit restrictions in the DFW corridor as data center "
                         "concentration increases competing demand for municipal water.",
        },
    },

    "pjm_va": {
        "regulatory": {
            "rating":    "Medium",
            "rationale": "Virginia's Clean Economy Act (VCEA) creates long-term clean "
                         "energy policy certainty but also imposes compliance costs on "
                         "utilities. State-level data center tax incentives have been "
                         "stable but are subject to legislative review.",
            "watch":     "Any revision to Virginia's data center sales tax exemption, "
                         "which is material to the economics of new development.",
        },
        "grid": {
            "rating":    "High",
            "rationale": "PJM's interconnection queue is the most backlogged in the "
                         "dataset at a 42-month median. The queue has worsened over the "
                         "past three years and PJM has not yet fully implemented FERC "
                         "Order 2023 reforms. Northern Virginia's transmission network "
                         "is approaching capacity in several zones.",
            "watch":     "Transmission capacity studies showing constrained import "
                         "capability into the Northern Virginia load pocket, which would "
                         "effectively cap new load additions.",
        },
        "execution": {
            "rating":    "High",
            "rationale": "Available land for large-format development is nearly exhausted "
                         "in Prince William and Loudoun Counties. Remaining sites carry "
                         "significant community opposition risk. Construction costs are "
                         "elevated due to concentrated demand for contractors.",
            "watch":     "County-level zoning decisions restricting new data center "
                         "development in Loudoun and Prince William Counties — this "
                         "regulatory risk has already materialized in part.",
        },
    },

    "pjm_oh": {
        "regulatory": {
            "rating":    "Medium",
            "rationale": "Ohio's RPS is weak (~8.5%) and the state legislature has "
                         "historically been ambivalent about renewable energy mandates. "
                         "This limits the policy tailwind but also limits downside risk "
                         "from overly aggressive clean energy requirements.",
            "watch":     "Any Ohio legislative action to further weaken the RPS, which "
                         "would reduce the available pool of clean energy contracts "
                         "for corporate buyers.",
        },
        "grid": {
            "rating":    "Medium",
            "rationale": "PJM queue backlog affects Ohio as it does the broader region, "
                         "but Ohio's transmission topology is less constrained than "
                         "Northern Virginia's. Queue times are improving modestly.",
            "watch":     "PJM capacity market outcomes — Ohio markets depend on capacity "
                         "payments for generation economics, and capacity price volatility "
                         "affects the energy cost outlook.",
        },
        "execution": {
            "rating":    "Low",
            "rationale": "Land is available and reasonably priced. Ohio's permitting "
                         "environment is faster than PJM's coastal markets. The Great "
                         "Lakes region provides reliable water access.",
            "watch":     "Local opposition to large industrial water users in communities "
                         "near Great Lakes tributaries, which has emerged as a concern "
                         "in some Michigan and Wisconsin markets.",
        },
    },

    "miso_ia": {
        "regulatory": {
            "rating":    "Low",
            "rationale": "Iowa has a long history of supporting wind energy development "
                         "with a stable regulatory environment. The state does not have "
                         "an aggressive RPS, but it also does not create significant "
                         "barriers to energy infrastructure development.",
            "watch":     "Federal transmission policy changes affecting MISO's long-range "
                         "transmission planning, which could alter the economics of "
                         "Iowa's wind-rich but transmission-dependent grid.",
        },
        "grid": {
            "rating":    "Low",
            "rationale": "MISO Iowa has a manageable interconnection queue, reliable "
                         "grid operations, and a generation mix that is increasingly "
                         "wind-dominated with strong system operator experience managing "
                         "variable generation.",
            "watch":     "MISO's transmission planning process and the timeline for "
                         "approved transmission projects that would improve export "
                         "capacity from Iowa wind zones.",
        },
        "execution": {
            "rating":    "Low",
            "rationale": "Low land costs, cooperative landowner relationships from decades "
                         "of wind development experience, and a straightforward permitting "
                         "environment. The main execution risk is fiber infrastructure "
                         "which requires capital investment in less-developed corridors.",
            "watch":     "Fiber construction costs and timelines in rural Iowa corridors "
                         "if a site is not already proximate to existing long-haul routes.",
        },
    },

    "spp_ks": {
        "regulatory": {
            "rating":    "Low",
            "rationale": "Kansas has a straightforward regulatory environment for energy "
                         "infrastructure with no significant permitting barriers for "
                         "wind or solar development. No state RPS creates less policy "
                         "uncertainty in either direction.",
            "watch":     "County-level setback requirements for wind turbines, which "
                         "have been enacted in some Kansas counties and could affect "
                         "site-specific development plans.",
        },
        "grid": {
            "rating":    "Low",
            "rationale": "SPP has the fastest interconnection queue in this dataset "
                         "alongside ERCOT. Kansas's grid is well-integrated with "
                         "the broader SPP footprint and has reliable operations.",
            "watch":     "SPP transmission congestion between Kansas wind zones and "
                         "load centers — curtailment risk is a known issue for wind "
                         "developers in some SPP corridors.",
        },
        "execution": {
            "rating":    "Medium",
            "rationale": "The primary execution risk in Kansas is not physical — it is "
                         "commercial. Without an anchor tenant, a developer is building "
                         "speculative infrastructure in a market with no existing "
                         "ecosystem. Land acquisition and construction are straightforward; "
                         "tenant acquisition is the uncertain variable.",
            "watch":     "The timeline to identify and close an anchor tenant commitment. "
                         "Without a committed customer, Kansas execution risk is "
                         "effectively the entire project.",
        },
    },

    "caiso_ca": {
        "regulatory": {
            "rating":    "Medium",
            "rationale": "California's policy environment is generally supportive of "
                         "clean energy but creates significant process risk through "
                         "CEQA review, which adds 12–24 months to major project "
                         "timelines. The regulatory environment is stable but slow.",
            "watch":     "Any expansion of CEQA exemptions for clean energy infrastructure, "
                         "which the California legislature has considered; or conversely, "
                         "any increase in CEQA scrutiny for data center water use.",
        },
        "grid": {
            "rating":    "High",
            "rationale": "CAISO has the longest interconnection queue in the dataset "
                         "(48-month median) and significant transmission congestion "
                         "in zones where new load would want to connect. Wildfire-driven "
                         "grid hardening requirements add cost and complexity.",
            "watch":     "CAISO's implementation of FERC Order 2023 interconnection "
                         "reforms — if California can accelerate queue processing, "
                         "the feasibility picture would improve meaningfully.",
        },
        "execution": {
            "rating":    "High",
            "rationale": "Land acquisition in California is the most expensive in the "
                         "dataset and is subject to CEQA challenges that can add years "
                         "to project timelines. Water availability is constrained and "
                         "faces increasing regulatory scrutiny as drought conditions "
                         "persist.",
            "watch":     "California state water board restrictions on industrial water "
                         "use, which have been proposed in various forms and could "
                         "materially affect cooling system design requirements.",
        },
    },

    "pnw_wa": {
        "regulatory": {
            "rating":    "Low",
            "rationale": "Washington has a strong and stable clean energy policy "
                         "environment including a 100% clean electricity standard. "
                         "Permitting for large industrial development is faster than "
                         "California and more predictable than the coastal PJM markets.",
            "watch":     "Any changes to Bonneville Power Administration's large power "
                         "contract terms, which affect industrial power pricing for "
                         "the largest customers in the region.",
        },
        "grid": {
            "rating":    "Low",
            "rationale": "Hydro-dominant grid with the best reliability metrics in the "
                         "dataset. NWPP interconnection queue is manageable. The region "
                         "has experience accommodating large industrial loads from "
                         "aluminum smelters and other heavy industry.",
            "watch":     "Long-run Columbia River hydro output under climate scenarios "
                         "with reduced snowpack. This is a 10–20 year risk, not "
                         "a near-term operational concern, but is relevant to "
                         "project financing assumptions.",
        },
        "execution": {
            "rating":    "Low",
            "rationale": "Water is abundant. Land is available in industrial corridors "
                         "outside the major metros. Construction costs are below California "
                         "and Northern Virginia. Utility relationships with Puget Sound "
                         "Energy and Pacific Power are well-established.",
            "watch":     "Land cost inflation if multiple large developers enter the "
                         "market simultaneously, which could erode the cost advantage "
                         "relative to ERCOT.",
        },
    },

    "se_ga": {
        "regulatory": {
            "rating":    "Low",
            "rationale": "Georgia is one of the most business-friendly states for "
                         "large industrial development. Georgia Power's regulated "
                         "structure provides rate certainty. State economic development "
                         "incentives for data centers are well-established.",
            "watch":     "Georgia Public Service Commission IRP decisions affecting "
                         "the pace of renewable energy additions to Georgia Power's "
                         "portfolio — relevant to the grid's clean energy content "
                         "and available clean power contracts.",
        },
        "grid": {
            "rating":    "Low",
            "rationale": "SERC reliability is strong. Georgia Power provides reliable "
                         "regulated utility service. Nuclear baseload (Vogtle) provides "
                         "firm clean power. The interconnection queue is more manageable "
                         "than PJM or CAISO.",
            "watch":     "Vogtle Unit 3 and 4 operational performance — these represent "
                         "a significant share of Georgia's clean baseload and any "
                         "operational issues would affect the grid's clean energy profile.",
        },
        "execution": {
            "rating":    "Low",
            "rationale": "Business-friendly permitting, available land at reasonable "
                         "cost, adequate water from Georgia's surface water system. "
                         "The main execution challenge is the limited renewable resource "
                         "quality, which requires sourcing clean power via PPA rather "
                         "than co-located generation.",
            "watch":     "PPA market depth in Georgia — if clean energy procurement "
                         "options are limited, tenants with strict clean energy "
                         "requirements may face sourcing challenges.",
        },
    },
}

# Integer mapping for sorting and comparison
_RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


# ── Risk matrix builder ───────────────────────────────────────────────────────

def risk_matrix(market_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Build a structured risk matrix: one row per market, three risk columns.

    Each cell contains the rating (Low / Medium / High).
    Also computes an aggregate_risk_level based on the worst dimension.
    """
    target = market_ids or list(RISK_REGISTRY.keys())
    rows = []

    for market_id in target:
        if market_id not in RISK_REGISTRY:
            continue
        entry = RISK_REGISTRY[market_id]
        reg_rating  = entry["regulatory"]["rating"]
        grid_rating = entry["grid"]["rating"]
        exec_rating = entry["execution"]["rating"]

        worst = max(
            [reg_rating, grid_rating, exec_rating],
            key=lambda r: _RISK_ORDER[r]
        )

        rows.append({
            "market_id":        market_id,
            "regulatory_risk":  reg_rating,
            "grid_risk":        grid_rating,
            "execution_risk":   exec_rating,
            "aggregate_risk":   worst,
        })

    return pd.DataFrame(rows)


def risk_rationales(market_id: str) -> pd.DataFrame:
    """
    Return a detailed risk breakdown for a single market including rationale
    and watch conditions for each dimension.
    """
    if market_id not in RISK_REGISTRY:
        raise ValueError(f"No risk data for market '{market_id}'.")

    entry = RISK_REGISTRY[market_id]
    rows = []
    for dimension in ("regulatory", "grid", "execution"):
        d = entry[dimension]
        rows.append({
            "dimension": dimension.title(),
            "rating":    d["rating"],
            "rationale": d["rationale"],
            "watch":     d["watch"],
        })
    return pd.DataFrame(rows)


# ── Opportunity vs. risk flag ─────────────────────────────────────────────────

def flag_opportunity_risk_quadrants(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each market into one of four quadrants:

        High Opportunity / Low Risk   → Priority — pursue actively
        High Opportunity / High Risk  → Proceed with caution — additional diligence required
        Low Opportunity  / Low Risk   → Monitor — revisit if conditions change
        Low Opportunity  / High Risk  → Avoid for now

    "High opportunity" = top half of the opportunity score distribution.
    "High risk" = aggregate risk rating of High on at least one dimension.

    Returns the scored DataFrame with two new columns:
        aggregate_risk, opportunity_risk_quadrant.
    """
    risk_df = risk_matrix()
    result = scored_df.merge(
        risk_df[["market_id", "aggregate_risk"]],
        on="market_id",
        how="left",
    )

    median_score = result["opportunity_score"].median()

    def _quadrant(row) -> str:
        high_opp  = row["opportunity_score"] >= median_score
        high_risk = row["aggregate_risk"] == "High"
        if high_opp and not high_risk:
            return "High Opportunity / Low Risk"
        if high_opp and high_risk:
            return "High Opportunity / High Risk"
        if not high_opp and not high_risk:
            return "Low Opportunity / Low Risk"
        return "Low Opportunity / High Risk"

    result["opportunity_risk_quadrant"] = result.apply(_quadrant, axis=1)
    return result
