# src/ingest_from_pdfs.py
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Callable

import pandas as pd

# Prefer pypdf (pure python). If missing: pip install pypdf
try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "extracted"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"


# -----------------------------
# Data structure
# -----------------------------
@dataclass(frozen=True)
class MetricRow:
    company_id: str
    metric_id: str
    fiscal_year: int
    value: Optional[float]
    unit: str
    boundary: str
    scope: str
    method_note: str
    quality_flag: str
    source_id: str
    extraction_note: str


CANONICAL_COLUMNS = [
    "company_id",
    "metric_id",
    "fiscal_year",
    "value",
    "unit",
    "boundary",
    "scope",
    "method_note",
    "quality_flag",
    "source_id",
    "extraction_note",
]


# -----------------------------
# PDF helpers
# -----------------------------
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _read_pdf_pages_text(pdf_path: Path) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("Missing dependency: pypdf. Install with: pip install pypdf")

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(_clean_text(txt))
    return pages


def _find_first_page(pages: List[str], pattern: str, flags=re.IGNORECASE) -> Optional[int]:
    for i, t in enumerate(pages):
        if re.search(pattern, t, flags):
            return i + 1  # 1-indexed
    return None


def _find_number_near(
    text: str,
    anchor_pat: str,
    num_pat: str,
    window: int = 260,
    flags=re.IGNORECASE,
) -> Optional[str]:
    m = re.search(anchor_pat, text, flags=flags)
    if not m:
        return None
    w = text[m.end() : m.end() + window]
    n = re.search(num_pat, w, flags=flags)
    if not n:
        return None
    return n.group(1)


def _to_float(num_str: Optional[str]) -> Optional[float]:
    if num_str is None:
        return None
    s = num_str.replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def _find_pdf(filename: str) -> Optional[Path]:
    """
    Try best-effort to locate a PDF in:
      - data/raw/pdfs/
      - project root
      - anywhere under project root (recursive)
    """
    candidates = [
        PDF_DIR / filename,
        PROJECT_ROOT / filename,
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Recursive exact match
    for p in PROJECT_ROOT.rglob(filename):
        if p.is_file():
            return p

    # Recursive case-insensitive match fallback
    target = filename.lower()
    for p in PROJECT_ROOT.rglob("*.pdf"):
        if p.name.lower() == target:
            return p

    return None


# -----------------------------
# Extraction: Google 2024 Environmental Report
# -----------------------------
def extract_google_2024(pdf_path: Path) -> List[MetricRow]:
    """
    Conservative extraction with sanity bounds.
    Extracts:
      - pue (expects ~1.10)
      - renewable_share_pct proxy using reported ~64% carbon-free energy
      - disclosure binaries = 1
      - third_party_assurance_level = 0 unless explicitly detected
      - offset_share_scope2 proxy = 100 (annual matching language)
      - scope2_intensity computed only if BOTH:
            electricity_twh plausible AND scope2_market_tco2e plausible
        otherwise: leave intensity = None
    """
    fiscal_year = 2024
    source_id = "google_2024_env_01"

    pages = _read_pdf_pages_text(pdf_path)

    # PUE
    pue_page = _find_first_page(
        pages,
        r"\bPUE\b.*\b1\.10\b|average annual power usage effectiveness.*\b1\.10\b",
    )
    pue_val = 1.10 if pue_page else None

    # CFE% (~64%)
    cfe_page = _find_first_page(
        pages,
        r"approximately\s+64%\s+carbon-free\s+energy|\b64%\b.*carbon-free energy",
    )
    cfe_val = 64.0 if cfe_page else None

    # Electricity consumption (TWh)
    elec_twh = None
    elec_page = None
    for i, t in enumerate(pages, start=1):
        if re.search(r"Total electricity consumption", t, flags=re.IGNORECASE):
            num = _find_number_near(t, r"Total electricity consumption", r"([0-9]+(?:\.[0-9]+)?)\s*TWh")
            if num is None:
                num = _find_number_near(t, r"Total electricity consumption", r"([0-9]+(?:\.[0-9]+)?)")
            v = _to_float(num)
            if v is not None:
                elec_twh = v
                elec_page = i
                break

    # Scope 2 (market-based) emissions (tCO2e)
    s2_tco2e = None
    s2_page = None
    for i, t in enumerate(pages, start=1):
        if re.search(r"Scope 2\s*\(market-based\)", t, flags=re.IGNORECASE):
            num = _find_number_near(t, r"Scope 2\s*\(market-based\)", r"([0-9]{1,3}(?:,[0-9]{3})+)")
            v = _to_float(num)
            if v is not None:
                s2_tco2e = v
                s2_page = i
                break

    # Sanity filters
    if s2_tco2e is not None and s2_tco2e < 1_000_000:
        s2_tco2e = None
    if elec_twh is not None and elec_twh > 50:
        elec_twh = None

    # Compute intensity if both plausible
    scope2_intensity = None
    if s2_tco2e is not None and elec_twh is not None:
        elec_mwh = elec_twh * 1_000_000.0
        scope2_intensity = s2_tco2e / elec_mwh

    rows: List[MetricRow] = []

    # Disclosure binaries
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="reports_data_center_metrics",
            fiscal_year=fiscal_year,
            value=1.0,
            unit="binary",
            boundary="company",
            scope="n_a",
            method_note="disclosed data center metrics",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=f"PUE/data center efficiency disclosed (page {pue_page})."
            if pue_page
            else "Data center metrics discussed (page not detected).",
        )
    )
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="reports_scope2_market_and_location",
            fiscal_year=fiscal_year,
            value=1.0,
            unit="binary",
            boundary="company",
            scope="n_a",
            method_note="disclosed market and location scope2",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Market vs location Scope 2 accounting discussed with figures/tables.",
        )
    )
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="reports_electricity_consumption",
            fiscal_year=fiscal_year,
            value=1.0,
            unit="binary",
            boundary="company",
            scope="n_a",
            method_note="disclosed electricity consumption",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=(
                f"Total electricity consumption parsed as {elec_twh} TWh (page {elec_page})."
                if elec_twh is not None and elec_page is not None
                else "Total electricity consumption referenced; parser did not capture a plausible TWh value."
            ),
        )
    )

    # Assurance (conservative default)
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="third_party_assurance_level",
            fiscal_year=fiscal_year,
            value=0.0,
            unit="level",
            boundary="company",
            scope="n_a",
            method_note="third-party assurance level (0 none / 1 limited / 2 reasonable)",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="No explicit third-party assurance statement detected in extracted text (v1 default = 0).",
        )
    )

    # Numeric metrics
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="pue",
            fiscal_year=fiscal_year,
            value=pue_val,
            unit="ratio",
            boundary="datacenters",
            scope="scope2",
            method_note="power usage effectiveness (PUE) for data centers",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=f"Average annual PUE reported as 1.10 (page {pue_page})."
            if pue_page
            else "Average annual PUE likely reported as 1.10; page not detected.",
        )
    )
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="renewable_share_pct",
            fiscal_year=fiscal_year,
            value=cfe_val,
            unit="pct",
            boundary="company",
            scope="scope2",
            method_note="share of electricity from renewable/carbon-free sources",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=f"Global average ~64% carbon-free energy (page {cfe_page})."
            if cfe_page
            else "Carbon-free energy share discussed; % not captured.",
        )
    )

    # Scope2 intensity
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="scope2_intensity",
            fiscal_year=fiscal_year,
            value=scope2_intensity,
            unit="tCO2_per_MWh",
            boundary="company",
            scope="scope2",
            method_note="Scope 2 market-based intensity = Scope 2 (market-based) / electricity consumption",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=(
                f"Computed using Scope 2 (market-based) {s2_tco2e:,.0f} tCO2e (page {s2_page}) and electricity {elec_twh} TWh (page {elec_page})."
                if scope2_intensity is not None
                else "Scope 2 intensity not computed (missing plausible Scope 2 market-based total and/or electricity consumption)."
            ),
        )
    )

    # Offset proxy
    rows.append(
        MetricRow(
            company_id="google",
            metric_id="offset_share_scope2",
            fiscal_year=fiscal_year,
            value=100.0,
            unit="pct",
            boundary="company",
            scope="scope2",
            method_note="proxy: annual renewable energy matching via contractual instruments",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Proxy set to 100 based on 100% annual renewable energy matching language in report.",
        )
    )

    return rows


# -----------------------------
# Extraction: Microsoft 2024 Environmental Sustainability Report
# -----------------------------
def extract_microsoft_2024(pdf_path: Path) -> List[MetricRow]:
    """
    Conservative extraction:
      - PUE design rating 1.12 (if detected)
      - reports_data_center_metrics = 1 if PUE detected
      - DO NOT mark electricity consumption as disclosed unless total electricity use is explicitly present
      - DO NOT fabricate renewable share, scope2 intensity, offset share
      - assurance defaults 0 unless explicit assurance statement detected
    """
    fiscal_year = 2024
    source_id = "microsoft_2024_sust_01"

    pages = _read_pdf_pages_text(pdf_path)

    # PUE 1.12 design rating
    pue_page = _find_first_page(
        pages,
        r"design rating of 1\.12 PUE|delivered a design rating of 1\.12|\b1\.12\b.*\bPUE\b",
    )
    pue_val = 1.12 if pue_page else None

    rows: List[MetricRow] = []

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="reports_data_center_metrics",
            fiscal_year=fiscal_year,
            value=1.0 if pue_page else 0.0,
            unit="binary",
            boundary="company",
            scope="n_a",
            method_note="disclosed data center metrics",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=f"PUE design rating disclosed (page {pue_page})."
            if pue_page
            else "Data center metrics not detected in extracted text.",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="reports_electricity_consumption",
            fiscal_year=fiscal_year,
            value=0.0,
            unit="binary",
            boundary="company",
            scope="n_a",
            method_note="disclosed electricity consumption",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Total electricity consumption not explicitly detected in extracted text (do not treat renewable use as total).",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="reports_scope2_market_and_location",
            fiscal_year=fiscal_year,
            value=0.0,
            unit="binary",
            boundary="company",
            scope="n_a",
            method_note="disclosed market and location scope2",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Market-based vs location-based Scope 2 totals not explicitly detected in extracted text.",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="third_party_assurance_level",
            fiscal_year=fiscal_year,
            value=0.0,
            unit="level",
            boundary="company",
            scope="n_a",
            method_note="third-party assurance level (0 none / 1 limited / 2 reasonable)",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="No explicit third-party assurance statement detected in extracted text (v1 default = 0).",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="pue",
            fiscal_year=fiscal_year,
            value=pue_val,
            unit="ratio",
            boundary="datacenters",
            scope="scope2",
            method_note="power usage effectiveness (PUE) for data centers",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note=f"Design PUE rating reported as 1.12 (page {pue_page})."
            if pue_page
            else "Design PUE rating not detected.",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="renewable_share_pct",
            fiscal_year=fiscal_year,
            value=None,
            unit="pct",
            boundary="company",
            scope="scope2",
            method_note="share of electricity from renewable/carbon-free sources",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Renewable electricity share not computable from this report text (needs % share or total + renewable).",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="scope2_intensity",
            fiscal_year=fiscal_year,
            value=None,
            unit="tCO2_per_MWh",
            boundary="company",
            scope="scope2",
            method_note="Scope 2 intensity (market/location per methodology)",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Scope 2 intensity not available from this report text; likely requires a separate ESG data table/fact sheet.",
        )
    )

    rows.append(
        MetricRow(
            company_id="microsoft",
            metric_id="offset_share_scope2",
            fiscal_year=fiscal_year,
            value=None,
            unit="pct",
            boundary="company",
            scope="scope2",
            method_note="share of scope 2 addressed via offsets/RECs (offset reliance proxy)",
            quality_flag="self_reported",
            source_id=source_id,
            extraction_note="Offset/REC reliance proxy not explicitly quantified in extracted text; leave null for v1.",
        )
    )

    return rows

def extract_amazon_2024(pdf_path: Path) -> List[MetricRow]:
    """
    Amazon 2024 Sustainability Report (covers 2024 metrics in the report tables).

    Conservative extraction (null > wrong):
      - Scope 2 (market-based) total emissions is disclosed in MMT CO2e -> convert to tCO2e
      - Renewable matching claim: 100% of electricity consumed matched with renewable sources (REC/PPA matching)
      - Electricity consumption total is NOT disclosed as a numeric total -> leave null and do not compute intensity
      - PUE is discussed conceptually but no numeric PUE -> leave null
      - Assurance statements exist; set third_party_assurance_level = 1 unless reasonable assurance is explicitly detected
    """
    fiscal_year = 2024
    source_id = "amazon_2024_sust_01"

    pages = _read_pdf_pages_text(pdf_path)

    # --- Scope 2 (market-based) in MMT CO2e ---
    # The data table shows: "Emissions from Purchased Electricity (Scope 2)* ... 2024 ... 2.80"
    # We attempt to parse the 2024 value, then convert MMT -> tCO2e.
    scope2_mmt = None
    scope2_page = None

    for i, t in enumerate(pages, start=1):
        if re.search(r"Emissions from Purchased Electricity\s*\(Scope 2\)", t, flags=re.IGNORECASE):
            # Try to capture the last (2024) value if the row is fully present
            # Pattern: "... Scope 2)* 5.50 5.27 4.07 3.06 2.76 2.80 ..."
            m = re.search(
                r"Emissions from Purchased Electricity\s*\(Scope 2\)\*?\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)",
                t,
                flags=re.IGNORECASE,
            )
            if m:
                scope2_mmt = _to_float(m.group(6))  # 2024 value
                scope2_page = i
                break

    scope2_tco2e = None
    if scope2_mmt is not None:
        # Convert million metric tons CO2e -> metric tons CO2e
        scope2_tco2e = scope2_mmt * 1_000_000.0

        # Sanity: reject absurdly small values
        if scope2_tco2e < 100_000:
            scope2_tco2e = None

    # --- Renewable matching: 100% electricity matched with renewables in 2024 ---
    ren_page = _find_first_page(
        pages,
        r"100%\s+of\s+electricity\s+consumed\s+by\s+Amazon\s+was\s+matched\s+with\s+renewable\s+energy\s+sources\s+in\s+2024",
    )
    renewable_share = 100.0 if ren_page else None

    # --- Electricity consumption total (NOT disclosed numerically) ---
    # We treat as not disclosed unless we find a numeric total electricity consumption (MWh/TWh).
    # (Keep conservative. Do not infer from renewable projects.)
    elec_total_found = False
    for t in pages:
        if re.search(r"Total electricity consumption\s+[0-9]", t, flags=re.IGNORECASE) and re.search(r"\b(MWh|GWh|TWh)\b", t, flags=re.IGNORECASE):
            elec_total_found = True
            break

    reports_elec = 1.0 if elec_total_found else 0.0

    # --- Scope 2 market vs location disclosure ---
    # This report explicitly notes Scope 2 is market-based in the table footnote. Treat as market-based disclosure present.
    reports_scope2 = 1.0 if scope2_tco2e is not None else 0.0

    # --- PUE numeric (not present; only conceptual) ---
    # Search for a likely numeric PUE pattern. If none, leave null.
    pue_page = _find_first_page(
        pages,
        r"\bPUE\b.*\b1\.[0-9]{2}\b|\b1\.[0-9]{2}\b.*\bPUE\b",
    )
    pue_val = None  # keep conservative; don't assume

    reports_dc_metrics = 1.0 if pue_page else 0.0

    # --- Assurance detection ---
    # The report includes an "Assurance Statements" section and links. Without the separate assurance PDFs, assume limited.
    assurance_page = _find_first_page(pages, r"Assurance Statements|assurance statements", flags=re.IGNORECASE)
    assurance_level = 1.0 if assurance_page else 0.0

    rows: List[MetricRow] = []

    rows.append(MetricRow(
        company_id="amazon", metric_id="reports_data_center_metrics", fiscal_year=fiscal_year, value=reports_dc_metrics,
        unit="binary", boundary="company", scope="n_a", method_note="disclosed data center metrics",
        quality_flag="self_reported", source_id=source_id,
        extraction_note="No numeric PUE/data center efficiency metric detected in report text." if not pue_page else f"Numeric PUE appears in report text (page {pue_page})."
    ))

    rows.append(MetricRow(
        company_id="amazon", metric_id="reports_electricity_consumption", fiscal_year=fiscal_year, value=reports_elec,
        unit="binary", boundary="company", scope="n_a", method_note="disclosed electricity consumption",
        quality_flag="self_reported", source_id=source_id,
        extraction_note="No numeric total electricity consumption (MWh/GWh/TWh) detected in report text; do not infer." if not elec_total_found else "Numeric total electricity consumption detected in report text."
    ))

    rows.append(MetricRow(
        company_id="amazon", metric_id="reports_scope2_market_and_location", fiscal_year=fiscal_year, value=reports_scope2,
        unit="binary", boundary="company", scope="n_a", method_note="disclosed market and location scope2",
        quality_flag="self_reported", source_id=source_id,
        extraction_note=f"Scope 2 purchased electricity disclosed in data table (page {scope2_page}); market-based method referenced in table footnote." if scope2_tco2e is not None else "Scope 2 purchased electricity total not reliably parsed from report text."
    ))

    rows.append(MetricRow(
        company_id="amazon", metric_id="third_party_assurance_level", fiscal_year=fiscal_year, value=assurance_level,
        unit="level", boundary="company", scope="n_a",
        method_note="third-party assurance level (0 none / 1 limited / 2 reasonable)",
        quality_flag="self_reported", source_id=source_id,
        extraction_note=f"Assurance statements section present (page {assurance_page}); set to 1 (limited) without separate assurance document verification." if assurance_page else "No assurance statements detected in extracted text (default = 0)."
    ))

    rows.append(MetricRow(
        company_id="amazon", metric_id="pue", fiscal_year=fiscal_year, value=pue_val,
        unit="ratio", boundary="datacenters", scope="scope2",
        method_note="power usage effectiveness (PUE) for data centers",
        quality_flag="self_reported", source_id=source_id,
        extraction_note="No numeric PUE reported in extracted report text (do not infer)." if not pue_page else f"Numeric PUE appears in text (page {pue_page}) but extraction not implemented (v1 conservative)."
    ))

    rows.append(MetricRow(
        company_id="amazon", metric_id="renewable_share_pct", fiscal_year=fiscal_year, value=renewable_share,
        unit="pct", boundary="company", scope="scope2",
        method_note="share of electricity from renewable/carbon-free sources",
        quality_flag="self_reported", source_id=source_id,
        extraction_note=f"Report states 100% of electricity consumed was matched with renewable energy sources in 2024 (page {ren_page})." if ren_page else "Renewable electricity matching claim not detected in extracted text."
    ))

    # We cannot compute intensity because we do not have electricity total
    rows.append(MetricRow(
        company_id="amazon", metric_id="scope2_intensity", fiscal_year=fiscal_year, value=None,
        unit="tCO2_per_MWh", boundary="company", scope="scope2",
        method_note="Scope 2 market-based intensity = Scope 2 (market-based) / electricity consumption",
        quality_flag="self_reported", source_id=source_id,
        extraction_note="Scope 2 intensity not computed: report does not disclose total electricity consumption as a numeric total."
    ))

    # Offset / REC matching proxy: if renewable matching is 100%, treat proxy = 100
    rows.append(MetricRow(
        company_id="amazon", metric_id="offset_share_scope2", fiscal_year=fiscal_year, value=100.0 if renewable_share == 100.0 else None,
        unit="pct", boundary="company", scope="scope2",
        method_note="proxy: renewable energy matching (REC/PPA) reliance for Scope 2",
        quality_flag="self_reported", source_id=source_id,
        extraction_note="Proxy set to 100 based on statement that 100% of electricity consumed was matched with renewable energy sources in 2024."
        if renewable_share == 100.0 else "No explicit 100% renewable matching statement detected; leave null."
    ))

    # Optional: store raw scope2 total as a separate metric only if you later add it to metric_definitions
    # For now: do NOT add new metric_id (keep your current schema).

    return rows
# -----------------------------
# Registry: add new companies here
# -----------------------------
ExtractorFn = Callable[[Path], List[MetricRow]]

EXTRACTORS: Dict[str, ExtractorFn] = {
    "google": extract_google_2024,
    "microsoft": extract_microsoft_2024,
    "amazon": extract_amazon_2024,
}

DEFAULT_FILENAMES: Dict[str, str] = {
    "google": "google-2024-environmental-report.pdf",
    "microsoft": "Microsoft-2024-Environmental-Sustainability-Report.pdf",
    "amazon": "2024-amazon-sustainability-report.pdf",
}


# -----------------------------
# Upsert helpers
# -----------------------------
def rows_to_df(rows: List[MetricRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    # Ensure all canonical columns exist
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[CANONICAL_COLUMNS]


def _ensure_metrics_file(metrics_path: Path) -> None:
    """
    If metrics_raw.csv doesn't exist yet, create it with canonical columns.
    (Your repo already has it — this is just safety.)
    """
    if metrics_path.exists():
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_COLUMNS).to_csv(metrics_path, index=False)


def upsert_metrics(metrics_path: Path, new_rows: pd.DataFrame) -> None:
    """
    True upsert:
      - key: (company_id, metric_id, fiscal_year)
      - updates existing rows on key match
      - appends new rows when key not present
    """
    _ensure_metrics_file(metrics_path)
    existing = pd.read_csv(metrics_path)

    # Ensure canonical schema for existing
    for col in CANONICAL_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
    existing = existing[CANONICAL_COLUMNS]

    # Ensure canonical schema for new
    for col in CANONICAL_COLUMNS:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    new_rows = new_rows[CANONICAL_COLUMNS].copy()

    key_cols = ["company_id", "metric_id", "fiscal_year"]

    # Standardize dtypes
    existing["company_id"] = existing["company_id"].astype(str)
    existing["metric_id"] = existing["metric_id"].astype(str)
    existing["fiscal_year"] = existing["fiscal_year"].astype("Int64")

    new_rows["company_id"] = new_rows["company_id"].astype(str)
    new_rows["metric_id"] = new_rows["metric_id"].astype(str)
    new_rows["fiscal_year"] = new_rows["fiscal_year"].astype("Int64")

    # Upsert via index
    existing_idx = existing.set_index(key_cols)
    new_idx = new_rows.set_index(key_cols)

    # Update overlapping keys
    existing_idx.update(new_idx)

    # Append missing keys
    missing_keys = new_idx.index.difference(existing_idx.index)
    if len(missing_keys) > 0:
        existing_idx = pd.concat([existing_idx, new_idx.loc[missing_keys]], axis=0)

    out = existing_idx.reset_index()

    # Keep stable ordering (nice for diffs)
    out = out.sort_values(["company_id", "metric_id", "fiscal_year"], kind="mergesort")
    out = out[CANONICAL_COLUMNS]
    out.to_csv(metrics_path, index=False)


# -----------------------------
# CLI / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AISRI PDF ingestion -> metrics_raw.csv (conservative extraction, true upsert)."
    )
    parser.add_argument(
        "--company",
        type=str,
        default="all",
        help="Company id to ingest (e.g., google, microsoft) or 'all'.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Optional explicit path to PDF. If omitted, uses default filename search.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=str(RAW_DIR / "metrics_raw.csv"),
        help="Path to metrics_raw.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_path)

    # Determine companies to run
    if args.company.lower() == "all":
        companies = sorted(EXTRACTORS.keys())
    else:
        c = args.company.lower().strip()
        if c not in EXTRACTORS:
            raise ValueError(
                f"Unknown company '{c}'. Known: {sorted(EXTRACTORS.keys())}. "
                f"Add an extractor in EXTRACTORS to support it."
            )
        companies = [c]

    all_rows: List[MetricRow] = []

    for company_id in companies:
        extractor = EXTRACTORS[company_id]

        # Determine PDF path
        if args.pdf:
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                raise FileNotFoundError(f"--pdf path not found: {pdf_path}")
        else:
            default_name = DEFAULT_FILENAMES.get(company_id)
            if not default_name:
                print(f"[WARN] No default filename configured for '{company_id}'. Use --pdf to pass one.")
                continue
            pdf_path = _find_pdf(default_name)  # best-effort search
            if pdf_path is None:
                print(
                    f"[WARN] PDF not found for '{company_id}'. Expected filename '{default_name}'. "
                    f"Put it in data/raw/pdfs/ or pass --pdf."
                )
                continue

        print(f"[INFO] Ingesting {company_id} from: {pdf_path}")
        rows = extractor(pdf_path)

        if not rows:
            print(f"[WARN] Extractor returned 0 rows for '{company_id}'.")
            continue

        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No rows extracted; nothing to write.")
        return

    new_df = rows_to_df(all_rows)
    upsert_metrics(metrics_path, new_df)

    print("[OK] Upserted metrics for:", sorted(set(new_df["company_id"].tolist())))
    print(new_df[["company_id", "metric_id", "fiscal_year", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()