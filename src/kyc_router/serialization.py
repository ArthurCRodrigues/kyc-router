"""Frozen seven-field Donut sequence protocol, extracted from Phase 4 research.

No neural, dataset, reference, or correction dependency.
"""

import html
import re

from .schema import REQUIRED_FIELDS

TASK_TOKEN = "<s_alb_id>"
EOS_TOKEN = "</s>"
FIELD_PATTERN = re.compile(r"<s_([a-z_]+)>([^<>]*)</s_\1>")


def schema_tokens():
    return [TASK_TOKEN] + [tag for field in REQUIRED_FIELDS
                           for tag in (f"<s_{field}>", f"</s_{field}>")]


def serialize_fields(fields):
    """Serialize already-normalized values; preserve Unicode and punctuation."""
    if set(fields) != set(REQUIRED_FIELDS):
        raise ValueError("Targets require exactly the seven v1 fields.")
    if any(not isinstance(fields[f], str) or not fields[f] for f in REQUIRED_FIELDS):
        raise ValueError("Training targets require nonempty strings.")
    return TASK_TOKEN + "".join(
        f"<s_{field}>{html.escape(fields[field], quote=False)}</s_{field}>"
        for field in REQUIRED_FIELDS
    ) + EOS_TOKEN


def parse_sequence(sequence):
    """Parse the declared envelope and grammar, without guessing or correction.

    Any malformed/duplicate sequence returns fields=None: the shared scorer
    then counts all fields missing. A well-formed partial mapping retains its
    other fields; extra keys remain visible to the scorer as schema violations.
    """
    def failure(reason):
        return {"fields": None, "structural_failure": True, "errors": [reason]}

    if not isinstance(sequence, str):
        return failure("not_a_string")
    if not sequence.startswith(TASK_TOKEN):
        return failure("missing_task_token")
    if not sequence.endswith(EOS_TOKEN):
        return failure("missing_eos")
    body = sequence[len(TASK_TOKEN):-len(EOS_TOKEN)]
    fields, position = {}, 0
    while position < len(body):
        match = FIELD_PATTERN.match(body, position)
        if match is None:
            return failure("malformed_field_sequence")
        name, escaped = match.groups()
        if name in fields:
            return failure("duplicate_field")
        value = html.unescape(escaped)
        if html.escape(value, quote=False) != escaped:
            return failure("noncanonical_entity")
        fields[name] = value
        position = match.end()
    return {"fields": fields, "structural_failure": False, "errors": []}
