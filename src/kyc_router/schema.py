"""The seven-field v1 contract, in the established inference order."""

RAW_TO_CANONICAL = {
    "number": "document_number",
    "surname": "surname",
    "name": "given_names",
    "birth_date": "date_of_birth",
    "expiry_date": "expiry_date",
    "nationality": "nationality",
    "gender": "sex",
}
REQUIRED_FIELDS = tuple(RAW_TO_CANONICAL.values())
DATE_FIELDS = frozenset({"date_of_birth", "expiry_date"})
