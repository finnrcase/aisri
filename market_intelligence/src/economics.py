# market_intelligence/src/economics.py
#
# Project Economics Layer — Lightweight
#
# This module provides a structured framework for first-order project economics
# estimates. It is NOT a financial model. There are no DCF calculations, no
# capital structure assumptions, and no IRR outputs.
#
# What it does:
#   - Defines market-specific cost inputs at the level of granularity that is
#     supportable from public data (order-of-magnitude, not precise estimates)
#   - Computes a simple cost-revenue-payback framework for a reference project
#     size (default: 100 MW IT load)
#   - Outputs a structured summary table for comparison across markets
#
# The value of this layer is not precision. It is the demonstration that you
# understand which cost variables differ across markets and in which direction.
# Power cost, land cost, and interconnection cost are the three variables that
# vary most across markets at this level of analysis.
#
# Sources and methodology notes are embedded in the MARKET_ECONOMICS dict.

from __future__ import annotations

import pandas as pd


# ── Market economics inputs ───────────────────────────────────────────────────
#
# All figures are approximate and reflect 2023–2024 public data.
# Sources: JLL Data Center Cost Guide, CBRE Capital Markets,
#          EIA (power prices), LBNL (interconnection costs),
#          CoStar/USDA (land costs).
#
# construction_cost_per_mw:
#   Shell + MEP (mechanical, electrical, plumbing) for a wholesale
#   hyperscale data center campus. Range is $7–10M/MW depending on
#   power density requirements and market labor costs.
#   Source: JLL Data Center Outlook; Cushman & Wakefield cost studies.
#
# interconnection_cost_per_mw:
#   Estimated generator interconnection cost allocated per MW of load.
#   Ranges from ~$100k/MW (ERCOT, shallow queue) to ~$500k/MW (CAISO,
#   deep queue with extensive network upgrades). Source: LBNL.
#
# land_cost_per_acre:
#   Large-parcel industrial/agricultural land. Northern Virginia and
#   California are outliers on the high end. Source: USDA, CoStar.
#
# assumed_acres_per_mw:
#   Typical land footprint for a hyperscale campus including setbacks,
#   parking, and generation space. ~3–7 acres/MW depending on density.
#
# colocation_rate_mw_year:
#   Wholesale colocation/lease rate expressed as $/MW/year.
#   $100–150/kW/month for hyperscale wholesale = $1.2–1.8M/MW/year.
#   Markets with higher demand density command higher rates.
#   Source: CBRE, JLL market rate surveys.
#
# om_cost_per_mw_year:
#   Annual operating and maintenance cost per MW excluding power.
#   Includes staffing, maintenance contracts, insurance. ~$60–100k/MW/yr.
#
# pue:
#   Power Usage Effectiveness — ratio of total facility power to IT load.
#   1.4 is a reasonable estimate for a new build; hyperscalers achieve 1.1–1.2
#   but that requires significant investment in cooling technology.

MARKET_ECONOMICS: dict[str, dict] = {

    "ercot_tx": {
        "construction_cost_per_mw":    8_000_000,
        "interconnection_cost_per_mw":   200_000,
        "land_cost_per_acre":              8_000,
        "assumed_acres_per_mw":                5,
        "power_price_mwh":                    34,   # $/MWh from dataset
        "pue":                               1.40,
        "colocation_rate_mw_year":     1_500_000,   # $125/kW/month
        "om_cost_per_mw_year":            80_000,
        "notes": "DFW-area industrial land. Power price reflects ERCOT average.",
    },

    "pjm_va": {
        "construction_cost_per_mw":   10_000_000,   # elevated — labor competition
        "interconnection_cost_per_mw":   450_000,   # high — deep PJM queue, network upgrades
        "land_cost_per_acre":             50_000,   # Loudoun/Prince William premium
        "assumed_acres_per_mw":                4,   # denser — land constrained
        "power_price_mwh":                    42,
        "pue":                               1.35,  # hyperscale-grade efficiency in mature market
        "colocation_rate_mw_year":     2_000_000,   # $167/kW/month — premium market
        "om_cost_per_mw_year":           100_000,   # elevated labor costs
        "notes": "Land cost reflects Loudoun/Prince William County reality. Revenue premium reflects largest market depth.",
    },

    "pjm_oh": {
        "construction_cost_per_mw":    8_000_000,
        "interconnection_cost_per_mw":   300_000,
        "land_cost_per_acre":              5_000,
        "assumed_acres_per_mw":                5,
        "power_price_mwh":                    35,
        "pue":                               1.40,
        "colocation_rate_mw_year":     1_300_000,   # $108/kW/month — emerging market discount
        "om_cost_per_mw_year":            75_000,
        "notes": "Columbus/Central Ohio industrial corridor. Emerging market — revenue rate below established hubs.",
    },

    "miso_ia": {
        "construction_cost_per_mw":    7_500_000,   # lower labor costs
        "interconnection_cost_per_mw":   175_000,
        "land_cost_per_acre":              3_000,   # Iowa farmland
        "assumed_acres_per_mw":                6,   # more land per MW for co-located generation
        "power_price_mwh":                    27,
        "pue":                               1.40,
        "colocation_rate_mw_year":     1_200_000,   # $100/kW/month — less competitive demand
        "om_cost_per_mw_year":            65_000,
        "notes": "Cheap land and power but lower revenue ceiling due to limited existing demand.",
    },

    "spp_ks": {
        "construction_cost_per_mw":    7_000_000,   # lowest — rural labor market
        "interconnection_cost_per_mw":   150_000,
        "land_cost_per_acre":              2_000,   # Kansas agricultural land
        "assumed_acres_per_mw":                7,
        "power_price_mwh":                    26,
        "pue":                               1.40,
        "colocation_rate_mw_year":     1_100_000,   # $92/kW/month — greenfield discount
        "om_cost_per_mw_year":            60_000,
        "notes": "Lowest cost basis in dataset. Revenue assumption highly speculative — no established market rate.",
    },

    "caiso_ca": {
        "construction_cost_per_mw":   11_000_000,   # highest — CA labor and materials
        "interconnection_cost_per_mw":   500_000,   # extensive network upgrades
        "land_cost_per_acre":             75_000,   # Bay Area / LA industrial
        "assumed_acres_per_mw":                3,   # land-constrained, denser builds
        "power_price_mwh":                    58,
        "pue":                               1.35,
        "colocation_rate_mw_year":     2_200_000,   # $183/kW/month — highest demand market
        "om_cost_per_mw_year":           110_000,
        "notes": "Highest cost basis. Revenue premium partially offsets cost, but not enough for greenfield economics.",
    },

    "pnw_wa": {
        "construction_cost_per_mw":    8_500_000,
        "interconnection_cost_per_mw":   200_000,
        "land_cost_per_acre":              8_000,   # industrial corridor outside metro
        "assumed_acres_per_mw":                5,
        "power_price_mwh":                    29,
        "pue":                               1.35,  # hydro climate supports efficient cooling
        "colocation_rate_mw_year":     1_600_000,   # $133/kW/month
        "om_cost_per_mw_year":            80_000,
        "notes": "Cheap, clean, reliable power with a proven tenant base. Strong economics.",
    },

    "se_ga": {
        "construction_cost_per_mw":    8_000_000,
        "interconnection_cost_per_mw":   225_000,
        "land_cost_per_acre":              5_000,
        "assumed_acres_per_mw":                5,
        "power_price_mwh":                    37,
        "pue":                               1.40,
        "colocation_rate_mw_year":     1_350_000,   # $113/kW/month — improving market
        "om_cost_per_mw_year":            70_000,
        "notes": "Metro Atlanta corridor. Reasonable economics, improving demand story.",
    },
}


# ── Economics calculator ──────────────────────────────────────────────────────

def compute_project_economics(
    market_id: str,
    capacity_mw: float = 100.0,
) -> dict:
    """
    Compute first-order project economics for a reference-size development.

    Inputs:
        market_id    — market key from MARKET_ECONOMICS
        capacity_mw  — IT load capacity in MW (default 100 MW)

    Returns a dict with:
        total_capex             — total capital cost ($M)
        capex_per_mw            — $/MW all-in capital cost
        annual_power_cost       — $/year power operating cost
        annual_om_cost          — $/year non-power operating cost
        annual_total_opex       — $/year total operating cost
        annual_revenue          — $/year gross revenue at stated colocation rate
        annual_net_operating    — $/year revenue minus opex (pre-financing)
        simple_payback_years    — capex / annual_net_operating (undiscounted)

    IMPORTANT: This is a simplified screening-level estimate, not a financial
    model. It excludes: financing costs, depreciation, taxes, development fees,
    insurance beyond O&M, and revenue ramp-up assumptions. Simple payback
    is not a substitute for a full IRR analysis.
    """
    if market_id not in MARKET_ECONOMICS:
        raise ValueError(f"No economics data for market '{market_id}'.")

    e = MARKET_ECONOMICS[market_id]

    # Capital costs
    construction  = e["construction_cost_per_mw"] * capacity_mw
    interconnect  = e["interconnection_cost_per_mw"] * capacity_mw
    land          = e["land_cost_per_acre"] * e["assumed_acres_per_mw"] * capacity_mw
    total_capex   = construction + interconnect + land

    # Annual operating costs
    # Power cost: IT load × PUE × 8,760 hours × $/MWh
    annual_energy_mwh   = capacity_mw * e["pue"] * 8_760
    annual_power_cost   = annual_energy_mwh * e["power_price_mwh"]
    annual_om_cost      = e["om_cost_per_mw_year"] * capacity_mw
    annual_total_opex   = annual_power_cost + annual_om_cost

    # Revenue
    annual_revenue      = e["colocation_rate_mw_year"] * capacity_mw

    # Net operating income (pre-financing, pre-tax)
    annual_net_operating = annual_revenue - annual_total_opex

    # Simple payback (undiscounted — a rough proxy, not an IRR)
    simple_payback = (
        total_capex / annual_net_operating
        if annual_net_operating > 0
        else float("inf")
    )

    return {
        "market_id":             market_id,
        "capacity_mw":           capacity_mw,
        "total_capex_mm":        round(total_capex / 1e6, 1),
        "capex_per_mw_mm":       round(total_capex / capacity_mw / 1e6, 2),
        "annual_power_cost_mm":  round(annual_power_cost / 1e6, 1),
        "annual_om_cost_mm":     round(annual_om_cost / 1e6, 1),
        "annual_total_opex_mm":  round(annual_total_opex / 1e6, 1),
        "annual_revenue_mm":     round(annual_revenue / 1e6, 1),
        "annual_noi_mm":         round(annual_net_operating / 1e6, 1),
        "simple_payback_years":  round(simple_payback, 1),
        "notes":                 e["notes"],
    }


def economics_summary(capacity_mw: float = 100.0) -> pd.DataFrame:
    """
    Run compute_project_economics for all markets and return a summary table.

    Columns are formatted for readability: all dollar figures in $M,
    payback in years.
    """
    rows = [
        compute_project_economics(mid, capacity_mw)
        for mid in MARKET_ECONOMICS
    ]
    df = pd.DataFrame(rows)

    # Sort by simple payback (best economics first)
    df = df.sort_values("simple_payback_years").reset_index(drop=True)
    return df
