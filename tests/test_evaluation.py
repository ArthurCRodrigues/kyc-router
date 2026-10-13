"""Small, inspectable examples for the shared research metrics."""

import json
import unittest

import pandas as pd

from kyc_router.evaluation import levenshtein_edits, score_document, summarize_scores
from kyc_router.schema import REQUIRED_FIELDS


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = dict(zip(REQUIRED_FIELDS, (
            "123456789", "Ahmeti", "Agim", "15-10-1973", "07-08-2025",
            "Shqiptare/Albanian", "M",
        )))

    def scored_frames(self, prediction=None):
        rows, document = score_document(self.truth, self.truth if prediction is None else prediction)
        metadata = {"identity_key": "example/00", "source": "template"}
        return pd.DataFrame([{**metadata, **row} for row in rows]), pd.DataFrame([{**metadata, **document}])

    def test_character_edit_examples(self):
        self.assertEqual(levenshtein_edits("kitten", "sitting"), {
            "edit_distance": 3, "substitutions": 2, "deletions": 0, "insertions": 1,
        })
        self.assertEqual(levenshtein_edits("abc", "")["deletions"], 3)
        self.assertEqual(levenshtein_edits("", "abc")["insertions"], 3)
        self.assertEqual(levenshtein_edits("ë", "e")["edit_distance"], 1)
        self.assertEqual(levenshtein_edits("", "")["edit_distance"], 0)

    def test_normalized_match_retains_raw_difference(self):
        prediction = {**self.truth, "surname": " AHMETI ", "date_of_birth": "1973-10-15"}
        rows, document = score_document(self.truth, prediction)
        self.assertTrue(document["document_exact_match"])
        self.assertEqual(sum(row["raw_exact_match"] for row in rows), 5)
        self.assertEqual(sum(row["normalized_exact_match"] for row in rows), 7)
        self.assertEqual(sum(row["edit_distance"] for row in rows), 0)

    def test_missing_extra_and_invalid_structured_results(self):
        prediction = {**self.truth, "extra": "retained as a schema violation"}
        del prediction["surname"]
        rows, document = score_document(self.truth, prediction)
        self.assertEqual(document["missing_fields"], ["surname"])
        self.assertEqual(document["extra_fields"], ["extra"])
        self.assertFalse(document["schema_is_valid"])
        self.assertEqual(sum(row["normalized_exact_match"] for row in rows), 6)
        for malformed in ("{", "[]", None, 42):
            with self.subTest(malformed=malformed):
                rows, document = score_document(self.truth, malformed)
                self.assertTrue(document["structural_failure"])
                self.assertFalse(any(row["normalized_exact_match"] for row in rows))
        _, document = score_document(self.truth, json.dumps(self.truth))
        self.assertTrue(document["document_exact_match"])

    def test_valid_number_can_still_be_incorrect(self):
        rows, document = score_document(self.truth, {**self.truth, "document_number": "123456780"})
        number = next(row for row in rows if row["field"] == "document_number")
        self.assertTrue(number["prediction_is_valid"])
        self.assertFalse(number["normalized_exact_match"])
        self.assertFalse(document["document_exact_match"])

    def test_invalid_reference_is_not_silently_scored(self):
        for truth in ({**self.truth, "date_of_birth": "31-02-1973"}, {**self.truth, "surname": ""}, {}):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                score_document(truth, self.truth)

    def test_aggregate_counts_and_denominators(self):
        prediction = {**self.truth, "surname": "Ahmetx"}
        fields, documents = self.scored_frames(prediction)
        per_field, summary = summarize_scores(fields, documents)
        result = summary.iloc[0]
        self.assertEqual(len(per_field), 7)
        self.assertEqual(result["normalized_matches"], 6)
        self.assertAlmostEqual(result["macro_field_em"], 6 / 7)
        self.assertEqual(result["macro_field_em"], result["micro_field_em"])
        self.assertEqual(result["document_matches"], 0)
        self.assertEqual(result["edit_distance"], 1)
        self.assertAlmostEqual(result["cer"], 1 / fields["reference_characters"].sum())

    def test_aggregate_rejects_dropped_or_duplicate_rows(self):
        fields, documents = self.scored_frames()
        invalid_inputs = (
            (fields.iloc[:-1], documents),
            (pd.concat([fields, fields.iloc[:1]]), documents),
            (fields, pd.concat([documents, documents])),
            (fields.iloc[:0], documents.iloc[:0]),
        )
        for field_rows, document_rows in invalid_inputs:
            with self.subTest(rows=len(field_rows)), self.assertRaises(ValueError):
                summarize_scores(field_rows, document_rows)


if __name__ == "__main__":
    unittest.main()
