"""Established Phase 3 scoring, separate from inference and semantic checks.

Extracted from 04c/04d. Retain raw strings and normalized strings: an exact
match is not fuzzy matching, and a valid-looking value can still be wrong.
"""

from collections.abc import Mapping, Sequence
import json

import numpy as np
import pandas as pd

from .normalization import is_missing_scalar, normalize_field_value
from .schema import DATE_FIELDS, REQUIRED_FIELDS
from .validation import validate_field_value


def levenshtein_edits(reference: str, prediction: str) -> dict[str, int]:
    """Count character edits; ties prefer substitution, then deletion.

    The tie order preserves the notebook's diagnostic edit decomposition. The
    total distance is independent of which equally short edit path is selected.
"""
    n, m = len(reference), len(prediction)
    distances = np.zeros((n + 1, m + 1), dtype=np.int64)
    distances[:, 0] = np.arange(n + 1)
    distances[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distances[i, j] = min(
                distances[i - 1, j] + 1,
                distances[i, j - 1] + 1,
                distances[i - 1, j - 1] + (reference[i - 1] != prediction[j - 1]),
            )
    i, j = n, m
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    while i or j:
        if (
            i and j
            and distances[i, j]
            == distances[i - 1, j - 1] + (reference[i - 1] != prediction[j - 1])
        ):
            counts["substitutions"] += int(reference[i - 1] != prediction[j - 1])
            i, j = i - 1, j - 1
        elif i and distances[i, j] == distances[i - 1, j] + 1:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    return {"edit_distance": int(distances[n, m]), **counts}


def scoring_text(value: object) -> str:
    """Missing values score as an empty prediction, never as 'None' or 'nan'."""
    return "" if is_missing_scalar(value) else str(value)


def score_document(
    truth: Mapping[str, object],
    prediction: object,
    *,
    required_fields: Sequence[str] = REQUIRED_FIELDS,
) -> tuple[list[dict], dict]:
    """Score one mapping or JSON result without repairing missing/extra keys.

    Invalid structured output loses every required field. Invalid references
    raise an error: they must not silently change the metric denominator.
    """
    structural_failure = False
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except (ValueError, TypeError):
            prediction = None
    if not isinstance(prediction, Mapping):
        structural_failure = True
        prediction = {}
    expected = set(required_fields)
    if set(truth) != expected:
        raise ValueError("Ground-truth schema must equal the required fields.")
    missing = sorted(expected - set(prediction))
    extra = sorted(set(prediction) - expected)
    rows = []
    for field in required_fields:
        raw_truth = truth[field]
        raw_prediction = prediction.get(field)
        reference = scoring_text(normalize_field_value(field, raw_truth))
        hypothesis = scoring_text(normalize_field_value(field, raw_prediction))
        truth_validation = validate_field_value(field, reference)
        if not reference or not truth_validation["is_valid"]:
            raise ValueError(f"Invalid reference for {field}: {raw_truth!r}")
        validation = validate_field_value(field, hypothesis)
        date_is_valid = field not in DATE_FIELDS or validation["is_valid"]
        rows.append({
            "field": field,
            "raw_ground_truth": raw_truth,
            "raw_prediction": raw_prediction,
            "normalized_ground_truth": reference,
            "normalized_prediction": hypothesis,
            "raw_exact_match": not is_missing_scalar(raw_prediction) and raw_prediction == raw_truth,
            "normalized_exact_match": bool(hypothesis) and date_is_valid and hypothesis == reference,
            "reference_characters": len(reference),
            "missing_key": field in missing,
            "prediction_is_valid": validation["is_valid"],
            "validation_errors": list(validation["errors"]),
            **levenshtein_edits(reference, hypothesis),
        })
    return rows, {
        "structural_failure": structural_failure,
        "schema_is_valid": not structural_failure and not missing and not extra,
        "missing_fields": missing,
        "extra_fields": extra,
        "document_exact_match": all(row["normalized_exact_match"] for row in rows),
    }


def summarize_scores(
    field_scores: pd.DataFrame,
    document_scores: pd.DataFrame,
    *,
    required_fields: Sequence[str] = REQUIRED_FIELDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per source, rejecting duplicate or missing observations.

    Input rows must include identity_key/source, plus score_document's field
    and document outputs. Corpus CER pools edits and reference characters;
    macro Field EM gives each required field equal weight.
    """
    keys = ["identity_key", "source"]
    if field_scores.empty or document_scores.empty:
        raise ValueError("Cannot summarize an empty evaluation.")
    if field_scores.duplicated(keys + ["field"]).any():
        raise ValueError("Duplicate scored field observations.")
    if document_scores.duplicated(keys).any():
        raise ValueError("Duplicate scored document observations.")
    if set(field_scores["field"]) != set(required_fields):
        raise ValueError("Scored fields must equal the required fields.")
    expected_index = pd.MultiIndex.from_tuples(
        [(row.identity_key, row.source, field) for row in document_scores.itertuples()
         for field in required_fields], names=keys + ["field"])
    actual_index = pd.MultiIndex.from_frame(field_scores[keys + ["field"]])
    if set(actual_index) != set(expected_index):
        raise ValueError("Every scored document must contain exactly its required fields.")
    fields = field_scores.groupby(["source", "field"], as_index=False).agg(
        samples=("field", "size"), raw_matches=("raw_exact_match", "sum"),
        normalized_matches=("normalized_exact_match", "sum"),
        edit_distance=("edit_distance", "sum"), reference_characters=("reference_characters", "sum"),
        invalid_predictions=("prediction_is_valid", lambda x: (~x).sum()),
    )
    fields["raw_field_em"] = fields["raw_matches"] / fields["samples"]
    fields["field_em"] = fields["normalized_matches"] / fields["samples"]
    fields["cer"] = fields["edit_distance"] / fields["reference_characters"]
    summaries = []
    for source, group in fields.groupby("source", sort=True):
        docs = document_scores.loc[document_scores["source"] == source]
        summaries.append({
            "source": source, "documents": len(docs), "field_samples": int(group["samples"].sum()),
            "raw_matches": int(group["raw_matches"].sum()),
            "normalized_matches": int(group["normalized_matches"].sum()),
            "raw_field_em": group["raw_field_em"].mean(),
            "macro_field_em": group["field_em"].mean(),
            "micro_field_em": group["normalized_matches"].sum() / group["samples"].sum(),
            "edit_distance": int(group["edit_distance"].sum()),
            "reference_characters": int(group["reference_characters"].sum()),
            "cer": group["edit_distance"].sum() / group["reference_characters"].sum(),
            "document_matches": int(docs["document_exact_match"].sum()),
            "document_em": docs["document_exact_match"].mean(),
            "schema_failures": int((~docs["schema_is_valid"]).sum()),
        })
    return fields, pd.DataFrame(summaries)
