"""Structural failures must not silently repair or improve measured extraction."""

import unittest

from kyc_router.serialization import TASK_TOKEN, EOS_TOKEN, parse_sequence, serialize_fields
from kyc_router.evaluation import score_document
from kyc_router.normalization import normalize_document_values


TRUTH = normalize_document_values({
    "document_number": "391378355", "surname": "Ahmeti", "given_names": "Agim",
    "date_of_birth": "15-10-1973", "expiry_date": "07-08-2025",
    "nationality": "Shqiptare/Albanian", "sex": "M",
})


class SerializationTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_and_literal_delimiters(self):
        fields = {**TRUTH, "surname": "García & <literal> > O'Neil"}
        sequence = serialize_fields(fields)
        self.assertIn("&amp; &lt;literal&gt; &gt;", sequence)
        self.assertEqual(parse_sequence(sequence)["fields"], fields)

    def test_missing_and_extra_keys_keep_parseable_values(self):
        sequence = serialize_fields(TRUTH).replace("<s_sex>M</s_sex>", "<s_extra>value</s_extra>")
        parsed = parse_sequence(sequence)
        rows, doc = score_document(TRUTH, parsed["fields"])
        self.assertFalse(parsed["structural_failure"])
        self.assertEqual(sum(r["normalized_exact_match"] for r in rows), 6)
        self.assertEqual(doc["missing_fields"], ["sex"])
        self.assertEqual(doc["extra_fields"], ["extra"])
        self.assertFalse(doc["schema_is_valid"])

    def test_duplicate_malformed_and_truncated_sequences_lose_all_fields(self):
        good = serialize_fields(TRUTH)
        cases = [None, "", good[:-4], good + "junk", good.replace(TASK_TOKEN, "", 1),
                 good.replace("</s_sex>", "</s_surname>"),
                 good.replace(EOS_TOKEN, "<s_sex>F</s_sex>" + EOS_TOKEN),
                 good.replace("ahmeti", "a & b"), good.replace("ahmeti", "&quot;name&quot;")]
        for sequence in cases:
            with self.subTest(sequence=sequence):
                parsed = parse_sequence(sequence)
                self.assertTrue(parsed["structural_failure"])
                rows, doc = score_document(TRUTH, parsed["fields"])
                self.assertTrue(doc["structural_failure"])
                self.assertEqual(sum(r["normalized_exact_match"] for r in rows), 0)

    def test_empty_value_and_invalid_date_are_not_guessed(self):
        sequence = serialize_fields(TRUTH).replace("<s_sex>M</s_sex>", "<s_sex></s_sex>")
        sequence = sequence.replace("1973-10-15", "1973-02-30")
        parsed = parse_sequence(sequence)
        rows, _ = score_document(TRUTH, parsed["fields"])
        self.assertFalse(parsed["structural_failure"])
        self.assertEqual(parsed["fields"]["date_of_birth"], "1973-02-30")
        self.assertEqual(sum(r["normalized_exact_match"] for r in rows), 5)

    def test_field_order_does_not_change_value_correctness(self):
        sequence = TASK_TOKEN + "".join(f"<s_{k}>{v}</s_{k}>" for k, v in reversed(list(TRUTH.items()))) + EOS_TOKEN
        self.assertEqual(parse_sequence(sequence)["fields"], TRUTH)

    def test_empty_target_and_wrong_target_schema_are_rejected(self):
        for fields in [{}, {**TRUTH, "extra": "x"}, {**TRUTH, "surname": ""}]:
            with self.assertRaises(ValueError):
                serialize_fields(fields)


if __name__ == "__main__":
    unittest.main()
