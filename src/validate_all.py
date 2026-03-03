# src/validate_all.py
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH when running directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schema import (  # noqa: E402
    validate_companies,
    validate_sources,
    validate_metrics_raw,
)

# File paths
COMPANIES = ROOT / "data" / "raw" / "extracted" / "companies.csv"
SOURCES = ROOT / "data" / "raw" / "extracted" / "sources.csv"
METRIC_DEFS = ROOT / "data" / "raw" / "extracted" / "metric_definitions.csv"
METRICS_RAW = ROOT / "data" / "raw" / "extracted" / "metrics_raw.csv"


def main():
    print("Validating companies.csv...")
    validate_companies(COMPANIES)
    print("✓ companies.csv OK")

    print("Validating sources.csv...")
    validate_sources(SOURCES)
    print("✓ sources.csv OK")

    print("Validating metrics_raw.csv...")
    validate_metrics_raw(METRICS_RAW, METRIC_DEFS)
    print("✓ metrics_raw.csv OK")


if __name__ == "__main__":
    main()