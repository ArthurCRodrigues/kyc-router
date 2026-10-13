"""The shared field contract, independent of any model or dataset files."""

from datetime import date
import unittest

import numpy as np
import pandas as pd

from kyc_router.normalization import (
    is_missing_scalar,
    normalize_document_values,
    normalize_field_value,
    parse_canonical_date,
)
from kyc_router.schema import REQUIRED_FIELDS
from kyc_router.validation import validate_document_values, validate_field_value


def valid_fields():
    return {
        "document_number": "391378355",
        "surname": "Ahmeti",
        "given_names": "Agim",
        "date_of_birth": "15-10-1973",
        "expiry_date": "07-08-2025",
        "nationality": "Shqiptare/Albanian",
        "sex": "M",
    }


class NormalizationTests(unittest.TestCase):
    def test_canonical_field_order(self):
        self.assertEqual(REQUIRED_FIELDS, tuple(valid_fields()))

    def test_common_normalization_preserves_meaning(self):
        cases = [
            ("surname", "  AHMETI\t\n", "ahmeti"),
            ("given_names", "  Arbe\u0308rie  Marie\n", "arbërie marie"),
            ("surname", "O'NEILL-SMITH", "o'neill-smith"),
            ("surname", "Straße", "strasse"),
            ("nationality", " Shqiptare/Albanian ", "shqiptare/albanian"),
            ("document_number", " ab 12-3\t", "AB12-3"),
            ("sex", " f ", "F"),
            ("sex", "male", "MALE"),
            ("surname", 123, "123"),
        ]
        for field, raw, expected in cases:
            with self.subTest(field=field, raw=raw):
                self.assertEqual(normalize_field_value(field, raw), expected)
        self.assertNotEqual(
            normalize_field_value("given_names", "Arbërie"),
            normalize_field_value("given_names", "Arberie"),
        )

    def test_missing_and_empty_are_not_confused(self):
        for value in (None, float("nan"), np.float64("nan"), pd.NA, pd.NaT):
            with self.subTest(value=value):
                self.assertTrue(is_missing_scalar(value))
                self.assertIsNone(normalize_field_value("surname", value))
        for value in ("", "nan", 0, False):
            self.assertFalse(is_missing_scalar(value))
        self.assertFalse(is_missing_scalar(np.array([float("nan")])))
        self.assertEqual(normalize_field_value("surname", " \t"), "")

    def test_explicit_date_formats_only(self):
        for raw in ("29-02-2024", "29/02/2024", "29.02.2024", "2024-02-29"):
            self.assertEqual(normalize_field_value("date_of_birth", raw), "2024-02-29")
        for raw in ("29-02-2023", "31-04-2024", "1-2-2024", "02/29/2024", "2024/02/29"):
            self.assertEqual(normalize_field_value("expiry_date", raw), raw)
        self.assertEqual(parse_canonical_date("2024-02-29"), date(2024, 2, 29))
        for raw in (None, "29-02-2024", "2023-02-29", "2024-2-29"):
            self.assertIsNone(parse_canonical_date(raw))

    def test_document_schema_is_exact(self):
        raw = valid_fields()
        normalized = normalize_document_values(raw)
        self.assertEqual(tuple(normalized), REQUIRED_FIELDS)
        self.assertEqual(normalized["date_of_birth"], "1973-10-15")
        self.assertEqual(raw, valid_fields(), "Normalization must not mutate caller data.")
        with self.assertRaises(ValueError):
            normalize_document_values({k: v for k, v in raw.items() if k != "sex"})
        with self.assertRaises(ValueError):
            normalize_document_values({**raw, "confidence": "1"})
        with self.assertRaises(ValueError):
            normalize_field_value("unknown", "value")


class ValidationTests(unittest.TestCase):
    def test_valid_document(self):
        result = validate_document_values(normalize_document_values(valid_fields()))
        self.assertTrue(result["is_valid"])
        self.assertTrue(result["schema_is_valid"])
        self.assertEqual(result["document_errors"], ())
        self.assertEqual(result["missing_fields"], ())
        self.assertEqual(result["unexpected_fields"], ())

    def test_field_error_codes(self):
        cases = [
            ("surname", None, "missing_value"),
            ("surname", 123, "not_a_string"),
            ("surname", "", "empty_value"),
            ("date_of_birth", "2023-02-29", "invalid_date"),
            ("document_number", "39137835O", "invalid_document_number_shape"),
            ("document_number", "39137835", "invalid_document_number_shape"),
            ("nationality", "albanian", "unknown_nationality"),
            ("sex", "MALE", "invalid_sex_value"),
        ]
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                self.assertEqual(validate_field_value(field, value), {
                    "is_valid": False, "errors": (error,),
                })
        with self.assertRaises(ValueError):
            validate_field_value("unknown", "value")

    def test_date_order_and_schema_errors(self):
        fields = normalize_document_values(valid_fields())
        for expiry in (fields["date_of_birth"], "1970-01-01"):
            result = validate_document_values({**fields, "expiry_date": expiry})
            self.assertFalse(result["is_valid"])
            self.assertEqual(result["document_errors"], ("expiry_not_after_date_of_birth",))
        result = validate_document_values({k: v for k, v in fields.items() if k != "sex"})
        self.assertEqual(result["missing_fields"], ("sex",))
        self.assertFalse(result["schema_is_valid"])
        self.assertEqual(result["field_validations"]["sex"]["errors"], ("missing_value",))
        result = validate_document_values({**fields, "extra": "value"})
        self.assertEqual(result["unexpected_fields"], ("extra",))
        self.assertFalse(result["is_valid"])

    def test_validation_is_not_recognition_certification(self):
        fields = normalize_document_values(valid_fields())
        wrong_but_plausible = {**fields, "document_number": "391378356", "surname": "wrong"}
        original = dict(wrong_but_plausible)
        self.assertTrue(validate_document_values(wrong_but_plausible)["is_valid"])
        self.assertEqual(wrong_but_plausible, original)


if __name__ == "__main__":
    unittest.main()
