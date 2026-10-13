"""Shared v1 normalization for predictions and references; never OCR correction.

Extracted from 04b/04d. pandas' scalar missing-value behavior is deliberately
retained so existing annotation/evaluation callers keep identical semantics.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import re
import unicodedata

import numpy as np
import pandas as pd

from .schema import DATE_FIELDS, REQUIRED_FIELDS

_DATE_PATTERNS = (
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%d-%m-%Y"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),
    (re.compile(r"^\d{2}\.\d{2}\.\d{4}$"), "%d.%m.%Y"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
)
_CANONICAL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_missing_scalar(value: object) -> bool:
    """Recognize None/NaN/NA/NaT without treating an array as a scalar."""
    if value is None:
        return True
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def normalize_common_text(value: object) -> str | None:
    if is_missing_scalar(value):
        return None
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.split()).casefold()


def parse_canonical_date(value: object) -> date | None:
    if not isinstance(value, str) or not _CANONICAL_DATE.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_field_value(field_name: str, value: object) -> str | None:
    """Preserve malformed dates verbatim (after common normalization) for audit."""
    if field_name not in REQUIRED_FIELDS:
        raise ValueError(f"Unknown canonical field: {field_name!r}")
    text = normalize_common_text(value)
    if text is None:
        return None
    if field_name == "document_number":
        return "".join(text.split()).upper()
    if field_name == "sex":
        return text.upper()
    if field_name in DATE_FIELDS:
        for pattern, date_format in _DATE_PATTERNS:
            if pattern.fullmatch(text):
                try:
                    return datetime.strptime(text, date_format).date().isoformat()
                except ValueError:
                    return text
    return text


def normalize_document_values(
    field_values: Mapping[str, object],
    required_fields: Sequence[str] = REQUIRED_FIELDS,
) -> dict[str, str | None]:
    expected, received = set(required_fields), set(field_values)
    if expected != received:
        raise ValueError(
            f"Invalid document schema. Missing: {sorted(expected - received)}; "
            f"unexpected: {sorted(received - expected)}."
        )
    return {field: normalize_field_value(field, field_values[field]) for field in required_fields}
