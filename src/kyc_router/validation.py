"""Simple alb_id checks. Well-formed does not mean correctly recognized."""

from collections.abc import Mapping, Sequence
import re

from .normalization import is_missing_scalar, parse_canonical_date
from .schema import DATE_FIELDS, REQUIRED_FIELDS

_DOCUMENT_NUMBER = re.compile(r"^\d{9}$")
_NATIONALITIES = frozenset({"shqiptare/albanian"})
_SEX_VALUES = frozenset({"M", "F"})


def validate_field_value(field_name: str, normalized_value: object) -> dict:
    if field_name not in REQUIRED_FIELDS:
        raise ValueError(f"Unknown canonical field: {field_name!r}")
    errors = []
    if is_missing_scalar(normalized_value):
        errors.append("missing_value")
    elif not isinstance(normalized_value, str):
        errors.append("not_a_string")
    elif normalized_value == "":
        errors.append("empty_value")
    elif field_name in DATE_FIELDS:
        if parse_canonical_date(normalized_value) is None:
            errors.append("invalid_date")
    elif field_name == "document_number":
        if not _DOCUMENT_NUMBER.fullmatch(normalized_value):
            errors.append("invalid_document_number_shape")
    elif field_name == "nationality":
        if normalized_value not in _NATIONALITIES:
            errors.append("unknown_nationality")
    elif field_name == "sex":
        if normalized_value not in _SEX_VALUES:
            errors.append("invalid_sex_value")
    return {"is_valid": not errors, "errors": tuple(errors)}


def validate_document_values(
    normalized_values: Mapping[str, object],
    *,
    required_fields: Sequence[str] = REQUIRED_FIELDS,
) -> dict:
    expected, received = set(required_fields), set(normalized_values)
    missing, extra = tuple(sorted(expected - received)), tuple(sorted(received - expected))
    validations = {
        field: validate_field_value(field, normalized_values.get(field))
        for field in required_fields
    }
    errors = []
    birth = parse_canonical_date(normalized_values.get("date_of_birth"))
    expiry = parse_canonical_date(normalized_values.get("expiry_date"))
    if birth is not None and expiry is not None and expiry <= birth:
        errors.append("expiry_not_after_date_of_birth")
    schema_valid = not missing and not extra
    return {
        "is_valid": schema_valid and all(v["is_valid"] for v in validations.values()) and not errors,
        "schema_is_valid": schema_valid,
        "missing_fields": missing,
        "unexpected_fields": extra,
        "field_validations": validations,
        "document_errors": tuple(errors),
    }
