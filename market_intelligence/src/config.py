# market_intelligence/src/config.py
#
# Single source of truth for all scoring configuration.
# Edit pillar weights, metric weights, and directions here.
# The scoring pipeline reads from this file — no changes needed elsewhere.
#
# Convention: opportunity_score ranges 0–100 where 100 = maximum opportunity.
# This is the inverse of AISRI's risk convention.

from __future__ import annotations
from typing import Literal

Direction = Literal["higher_is_better", "lower_is_better"]


# ---------------------------------------------------------------------------
# Pillar weights
# Must sum to 1.0
# ---------------------------------------------------------------------------

PILLAR_WEIGHTS: dict[str, float] = {
    "demand":      0.30,
    "energy":      0.30,
    "feasibility": 0.25,
    "strategic":   0.15,
}


# ---------------------------------------------------------------------------
# Metric definitions
#
# Each entry defines:
#   pillar      — which pillar this metric rolls up into
#   weight      — share of that pillar's score (weights within a pillar sum to 1.0)
#   direction   — whether a higher raw value is better or worse
#   bounds      — (low, high) used for min-max normalization across the market set
#                 Set to None to derive bounds dynamically from the dataset.
#                 Explicit bounds are preferred where a natural scale exists.
#   description — what the variable measures
#   source      — where a real analyst would pull this data
#   unit        — unit of measurement for the raw value
# ---------------------------------------------------------------------------

METRICS: dict[str, dict] = {

    # -------------------------------------------------------------------------
    # Demand Potential
    # -------------------------------------------------------------------------
    "hyperscale_presence": {
        "pillar":       "demand",
        "weight":       0.30,
        "direction":    "higher_is_better",
        "bounds":       (0, 5),
        "description":  (
            "Ordinal score (0–5) for the number and scale of hyperscale cloud operators "
            "(AWS, Azure, Google Cloud, Meta) with existing or announced data center "
            "presence in the market. 5 = all four with multi-campus footprints."
        ),
        "source":       "CBRE Data Center Trends, Cushman & Wakefield Global DC Report",
        "unit":         "score 0–5",
    },

    "data_center_capacity_mw": {
        "pillar":       "demand",
        "weight":       0.35,
        "direction":    "higher_is_better",
        "bounds":       (0, 4000),
        "description":  (
            "Total installed and under-construction data center capacity in the market, "
            "in megawatts. A larger existing market signals proven demand and customer "
            "confidence, but may also indicate land / power constraints."
        ),
        "source":       "JLL Data Center Outlook, DC Byte market reports, CBRE",
        "unit":         "MW",
    },

    "tech_employment_index": {
        "pillar":       "demand",
        "weight":       0.20,
        "direction":    "higher_is_better",
        "bounds":       (0.5, 2.5),
        "description":  (
            "Tech sector employment in the market relative to the U.S. national average "
            "(1.0 = national average). Used as a proxy for local AI/software demand "
            "and talent availability for potential tenants."
        ),
        "source":       "U.S. Bureau of Labor Statistics (BLS) QCEW, CompTIA State of the Tech Workforce",
        "unit":         "index (1.0 = national avg)",
    },

    "fiber_connectivity_score": {
        "pillar":       "demand",
        "weight":       0.15,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) for long-haul and metro fiber density, internet exchange "
            "point presence, and latency to major U.S. population centers. Critical for "
            "AI inference workloads and hyperscale connectivity requirements."
        ),
        "source":       "FCC Broadband Data Collection, Telegeography fiber maps, PeeringDB",
        "unit":         "score 1–5",
    },

    # -------------------------------------------------------------------------
    # Energy Attractiveness
    # -------------------------------------------------------------------------
    "avg_wholesale_power_price_mwh": {
        "pillar":       "energy",
        "weight":       0.35,
        "direction":    "lower_is_better",
        "bounds":       (20, 65),
        "description":  (
            "Annual average day-ahead wholesale electricity price in the market's ISO/RTO, "
            "in $/MWh. Lower prices reduce operating costs for energy-intensive AI workloads. "
            "Values approximate 2023 annual averages; ERCOT, MISO, and SPP are structurally "
            "cheaper than PJM and CAISO."
        ),
        "source":       "EIA Electric Power Monthly, FERC LCIA price reports, ISO/RTO annual reports",
        "unit":         "$/MWh",
    },

    "renewable_capacity_factor_pct": {
        "pillar":       "energy",
        "weight":       0.30,
        "direction":    "higher_is_better",
        "bounds":       (20, 55),
        "description":  (
            "Weighted average capacity factor (%) for wind and solar resources available "
            "in or near the market. Higher capacity factors mean more energy output per "
            "dollar of capital invested in renewable generation — directly relevant to "
            "the economics of co-located or proximate clean energy projects."
        ),
        "source":       "NREL Annual Technology Baseline (ATB), NREL Wind and Solar resource maps",
        "unit":         "%",
    },

    "clean_energy_penetration_pct": {
        "pillar":       "energy",
        "weight":       0.20,
        "direction":    "higher_is_better",
        "bounds":       (10, 85),
        "description":  (
            "Share of total electricity generation from clean sources (wind, solar, hydro, "
            "nuclear) in the relevant grid region, as a percentage. Higher penetration "
            "reduces the carbon intensity of grid-delivered power and supports corporate "
            "sustainability commitments from data center tenants."
        ),
        "source":       "EIA Form 923 (generation by fuel type), EPA eGRID database",
        "unit":         "%",
    },

    "grid_reliability_score": {
        "pillar":       "energy",
        "weight":       0.15,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) for grid reliability based on published SAIDI/SAIFI outage "
            "metrics and ISO/RTO reliability assessments. ERCOT receives a deduction for "
            "its isolated interconnection and the 2021 weather event. CAISO receives a "
            "deduction for wildfire-related curtailment risk."
        ),
        "source":       "NERC Long-Term Reliability Assessment, EIA Electric Power Annual (outage data)",
        "unit":         "score 1–5",
    },

    # -------------------------------------------------------------------------
    # Deployment Feasibility
    # -------------------------------------------------------------------------
    "interconnection_queue_months": {
        "pillar":       "feasibility",
        "weight":       0.35,
        "direction":    "lower_is_better",
        "bounds":       (10, 55),
        "description":  (
            "Median estimated time in months to clear the generator interconnection queue "
            "in the relevant ISO/RTO. This is the single most common cause of project delay "
            "for new energy infrastructure. CAISO and PJM queues are severely backlogged; "
            "ERCOT and SPP are materially faster."
        ),
        "source":       "LBNL Interconnection Queue Report, ISO/RTO interconnection study timelines",
        "unit":         "months",
    },

    "permitting_speed_score": {
        "pillar":       "feasibility",
        "weight":       0.25,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) for the ease and speed of state and local permitting for "
            "large energy and data infrastructure projects. Reflects state regulatory climate, "
            "environmental review timelines, and local zoning flexibility. Texas and Iowa "
            "score highest; California scores lowest due to CEQA review requirements."
        ),
        "source":       "Industry surveys (Clean Energy States Alliance), state energy office reports",
        "unit":         "score 1–5",
    },

    "land_cost_index": {
        "pillar":       "feasibility",
        "weight":       0.20,
        "direction":    "lower_is_better",
        "bounds":       (1, 15),
        "description":  (
            "Relative land cost index for large-acreage industrial/agricultural land, where "
            "lower values indicate cheaper land. Kansas and Iowa (farmland) are cheapest; "
            "Northern Virginia and California are most expensive. Relevant because AI campuses "
            "and co-located generation require significant land area."
        ),
        "source":       "USDA Land Values Summary, CoStar industrial land comps, CBRE site selection data",
        "unit":         "index (relative, lower = cheaper)",
    },

    "water_availability_score": {
        "pillar":       "feasibility",
        "weight":       0.20,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) for freshwater availability for cooling. Data centers "
            "with evaporative cooling towers consume significant water. Pacific Northwest "
            "and Great Lakes markets score highest; California and western Kansas face "
            "water stress. Air-cooled designs reduce (but do not eliminate) this risk."
        ),
        "source":       "USGS Water Resources, WRI Aqueduct Water Risk Atlas, state water authority data",
        "unit":         "score 1–5",
    },

    # -------------------------------------------------------------------------
    # Strategic Fit
    # -------------------------------------------------------------------------
    "renewable_colocation_potential": {
        "pillar":       "strategic",
        "weight":       0.40,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) for the availability of developable land adjacent to or "
            "within the market for utility-scale wind or solar projects that could be "
            "co-located with or physically proximate to AI data center campuses. This is "
            "the core of Intersect Power's differentiated model."
        ),
        "source":       "NREL ReEDS model, NREL SolarAnywhere/WindNavigator, USDA NLCD land use maps",
        "unit":         "score 1–5",
    },

    "ppa_market_maturity_score": {
        "pillar":       "strategic",
        "weight":       0.30,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) for the depth and liquidity of the corporate Power Purchase "
            "Agreement (PPA) market in the state/region. A mature PPA market indicates "
            "willing counterparties (tech companies, industrials) and precedent transactions "
            "that reduce deal execution risk."
        ),
        "source":       "LevelTen Energy PPA Market Intelligence Report, BNEF Corporate PPA League Tables",
        "unit":         "score 1–5",
    },

    "state_policy_score": {
        "pillar":       "strategic",
        "weight":       0.30,
        "direction":    "higher_is_better",
        "bounds":       (1, 5),
        "description":  (
            "Ordinal score (1–5) reflecting the favorability of state energy and economic "
            "development policy for clean energy and data infrastructure. Includes state "
            "Renewable Portfolio Standards (RPS), data center sales tax exemptions, "
            "clean energy investment tax credits, and governor-level economic development "
            "prioritization."
        ),
        "source":       "DSIRE (Database of State Incentives for Renewables and Efficiency), "
                        "NCSL energy policy tracker, state economic development agency publications",
        "unit":         "score 1–5",
    },
}


# ---------------------------------------------------------------------------
# Derived convenience structure: pillar -> list of metric keys
# Used by the scoring engine to aggregate metrics into pillars.
# ---------------------------------------------------------------------------

PILLAR_METRICS: dict[str, list[str]] = {}
for _metric_id, _cfg in METRICS.items():
    _p = _cfg["pillar"]
    PILLAR_METRICS.setdefault(_p, []).append(_metric_id)


# ---------------------------------------------------------------------------
# Validation: assert weights sum correctly
# Runs at import time so misconfiguration is caught immediately.
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    pillar_sum = sum(PILLAR_WEIGHTS.values())
    assert abs(pillar_sum - 1.0) < 1e-6, (
        f"PILLAR_WEIGHTS must sum to 1.0, got {pillar_sum:.4f}"
    )

    for pillar, metric_ids in PILLAR_METRICS.items():
        metric_sum = sum(METRICS[m]["weight"] for m in metric_ids)
        assert abs(metric_sum - 1.0) < 1e-6, (
            f"Metric weights for pillar '{pillar}' must sum to 1.0, got {metric_sum:.4f}"
        )

    valid_pillars = set(PILLAR_WEIGHTS.keys())
    for metric_id, cfg in METRICS.items():
        assert cfg["pillar"] in valid_pillars, (
            f"Metric '{metric_id}' references unknown pillar '{cfg['pillar']}'"
        )
        assert cfg["direction"] in ("higher_is_better", "lower_is_better"), (
            f"Metric '{metric_id}' has invalid direction '{cfg['direction']}'"
        )


_validate_config()
