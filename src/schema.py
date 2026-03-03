# src/schema.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


# -------------------------
# Constants / allowed values
# -------------------------
Pillar = Literal["energy", "efficiency", "carbon", "offsets", "transparency"]
Direction = Literal["higher_is_better", "lower_is_better"]
Boundary = Literal["data_center", "compute", "company", "estimated", "unknown"]

Scope = Literal[
    "scope1",
    "scope2_mkt",
    "scope2_loc",
    "scope3",
    "n_a",
]

QualityFlag = Literal[
    "audited",
    "assured_limited",
    "assured_reasonable",
    "self_reported",
    "unclear",
]


# -------------------------
# Helper to load codebook
# -------------------------
@dataclass(frozen=True)
class Codebook:
    metric_ids: Set[str]
    pillars: Set[str]

    @staticmethod
    def load(metric_definitions_path: str | Path) -> "Codebook":
        path = Path(metric_definitions_path)
        if not path.exists():
            raise FileNotFoundError(f"metric_definitions.csv not found at: {path}")

        df = pd.read_csv(path)

        required_cols = {
            "metric_id",
            "pillar",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"metric_definitions.csv missing columns: {sorted(missing)}")

        metric_ids = set(df["metric_id"].astype(str).str.strip())
        pillars = set(df["pillar"].astype(str).str.strip())
        return Codebook(metric_ids=metric_ids, pillars=pillars)


# -------------------------
# Row schemas (raw tables)
# -------------------------
class MetricsRawRow(BaseModel):
    company_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    fiscal_year: int = Field(ge=2000, le=2100)

    value: str | int | float = Field(...)
    unit: str = Field(min_length=1)

    @field_validator("value", mode="before")
    @classmethod
    def value_not_blank(cls, v):
        # Allow numbers or strings
        try:
            if v != v:  # NaN check
                raise ValueError("value cannot be NaN")
        except Exception:
            pass

        v2 = str(v).strip()
        if v2 == "":
            raise ValueError("value cannot be blank")
        return v2


class CompaniesRow(BaseModel):
    company_id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    hq_country: str = Field(min_length=1)
    company_type: str = Field(min_length=1)
    public_private: str = Field(min_length=1)
    notes_v1_scope: Optional[str] = None
    last_updated_date: str = Field(min_length=1)

    @field_validator("company_id", "company_name")
    @classmethod
    def strip_company(cls, v: str) -> str:
        return v.strip()


class SourcesRow(BaseModel):
    source_id: str = Field(min_length=1)
    company_id: Optional[str] = None
    source_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    publish_date: str = Field(min_length=1)
    url: str = Field(min_length=1)
    accessed_date: str = Field(min_length=1)
    archive_url: Optional[str] = None
    citation_text: Optional[str] = None

    @field_validator("company_id", "archive_url", "citation_text", mode="before")
    @classmethod
    def nan_to_none(cls, v):
        # pandas uses NaN (float) for empty cells; convert those to None
        try:
            # NaN is the only value where v != v is True
            if v != v:
                return None
        except Exception:
            pass
        return v

    @field_validator("source_id", "source_type", "publisher")
    @classmethod
    def strip_source(cls, v: str) -> str:
        return v.strip()


# -------------------------
# Table-level validation
# -------------------------
def validate_metrics_raw(metrics_path: str | Path, metric_definitions_path: str | Path) -> None:
    """
    Raises ValueError with readable messages if validation fails.
    """
    codebook = Codebook.load(metric_definitions_path)
    df = pd.read_csv(metrics_path)

    required_cols = set(MetricsRawRow.model_fields.keys())
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metrics_raw.csv missing columns: {sorted(missing)}")

    errors = []
    for i, row in df.iterrows():
        try:
            parsed = MetricsRawRow(**row.to_dict())
            if parsed.metric_id not in codebook.metric_ids:
                raise ValueError(
                    f"Unknown metric_id '{parsed.metric_id}'. Must be one of codebook metric_ids."
                )
        except (ValidationError, ValueError) as e:
            errors.append(f"Row {i+2}: {e}")  # +2 accounts for header row and 0-index

    if errors:
        msg = "metrics_raw.csv failed validation:\n" + "\n".join(errors[:50])
        if len(errors) > 50:
            msg += f"\n...and {len(errors)-50} more errors"
        raise ValueError(msg)


def validate_companies(companies_path: str | Path) -> None:
    df = pd.read_csv(companies_path)
    required_cols = set(CompaniesRow.model_fields.keys())
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"companies.csv missing columns: {sorted(missing)}")

    errors = []
    for i, row in df.iterrows():
        try:
            CompaniesRow(**row.to_dict())
        except ValidationError as e:
            errors.append(f"Row {i+2}: {e}")

    if errors:
        raise ValueError("companies.csv failed validation:\n" + "\n".join(errors[:50]))


def validate_sources(sources_path: str | Path) -> None:
    df = pd.read_csv(sources_path)

    df = df.where(pd.notnull(df), None)

    # Convert NaN to None so optional fields validate correctly
    df = df.where(pd.notnull(df), None)

    required_cols = set(SourcesRow.model_fields.keys())
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"sources.csv missing columns: {sorted(missing)}")

    errors = []
    for i, row in df.iterrows():
        try:
            SourcesRow(**row.to_dict())
        except ValidationError as e:
            errors.append(f"Row {i+2}: {e}")

    if errors:
        raise ValueError("sources.csv failed validation:\n" + "\n".join(errors[:50]))
    
    df = pd.read_csv(sources_path)
    required_cols = set(SourcesRow.model_fields.keys())
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"sources.csv missing columns: {sorted(missing)}")

    errors = []
    for i, row in df.iterrows():
        try:
            SourcesRow(**row.to_dict())
        except ValidationError as e:
            errors.append(f"Row {i+2}: {e}")

    if errors:
        raise ValueError("sources.csv failed validation:\n" + "\n".join(errors[:50]))
        