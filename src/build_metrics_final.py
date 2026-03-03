# src/build_metrics_final.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_EXTRACTED = PROJECT_ROOT / "data" / "raw" / "extracted"
RAW_MANUAL = PROJECT_ROOT / "data" / "raw" / "manual"
PROCESSED = PROJECT_ROOT / "data" / "processed"

RAW_METRICS_PATH = RAW_EXTRACTED / "metrics_raw.csv"
MANUAL_METRICS_PATH = RAW_MANUAL / "metrics_manual.csv"
FINAL_METRICS_PATH = PROCESSED / "metrics_final.csv"

KEY_COLS = ["company_id", "metric_id", "fiscal_year"]

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

OPTIONAL_AUDIT_COLUMNS = ["page_ref", "verbatim_snippet"]  # manual only


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS + OPTIONAL_AUDIT_COLUMNS)
    return pd.read_csv(path)


def _coerce_schema(df: pd.DataFrame, include_audit: bool = False) -> pd.DataFrame:
    cols = CANONICAL_COLUMNS + (OPTIONAL_AUDIT_COLUMNS if include_audit else [])
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols].copy()

    # Normalize types
    df["company_id"] = df["company_id"].astype(str)
    df["metric_id"] = df["metric_id"].astype(str)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def build_metrics_final() -> None:
    if not RAW_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing extractor output: {RAW_METRICS_PATH}")

    extracted = _coerce_schema(_read_csv_if_exists(RAW_METRICS_PATH), include_audit=False)
    manual = _coerce_schema(_read_csv_if_exists(MANUAL_METRICS_PATH), include_audit=True)

    # If no manual rows, just copy extracted -> final
    if manual.empty:
        _ensure_dir(PROCESSED)
        extracted.to_csv(FINAL_METRICS_PATH, index=False)
        print(f"[OK] Built metrics_final.csv (no manual rows). Wrote: {FINAL_METRICS_PATH}")
        return

    # Validate manual keys
    for k in KEY_COLS:
        if manual[k].isna().any():
            bad = manual[manual[k].isna()]
            raise ValueError(
                "metrics_manual.csv has missing key fields. Fix these rows:\n"
                + bad[KEY_COLS + ["metric_id", "source_id"]].to_string(index=False)
            )

    # Deduplicate manual rows by key: last row wins
    manual = manual.sort_values(KEY_COLS, kind="mergesort").drop_duplicates(subset=KEY_COLS, keep="last")

    # Set indices for upsert
    ext_idx = extracted.set_index(KEY_COLS)
    man_idx = manual.set_index(KEY_COLS)

    # Only update/append the NON-key canonical columns (keys live in the index)
    NONKEY_CANON_COLS = [c for c in CANONICAL_COLUMNS if c not in KEY_COLS]
    man_canon = man_idx[NONKEY_CANON_COLS].copy()

    # 1) Update overlapping keys
    ext_idx.update(man_canon)

    # 2) Append missing keys
    missing_keys = man_canon.index.difference(ext_idx.index)
    if len(missing_keys) > 0:
        ext_idx = pd.concat([ext_idx, man_canon.loc[missing_keys]], axis=0)

    final = ext_idx.reset_index()
    final = final.sort_values(KEY_COLS, kind="mergesort")
    final = final[CANONICAL_COLUMNS]

    _ensure_dir(PROCESSED)
    final.to_csv(FINAL_METRICS_PATH, index=False)

    print("[OK] Built metrics_final.csv")
    print(f" - Extracted rows: {len(extracted):,}")
    print(f" - Manual rows:    {len(manual):,}")
    print(f" - Final rows:     {len(final):,}")
    print(f"Wrote: {FINAL_METRICS_PATH}")


if __name__ == "__main__":
    build_metrics_final()