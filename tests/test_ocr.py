"""Model-free request tests exercise the same torch/device boundaries as TrOCR."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kyc_router.ocr import OcrPipeline, load_pipeline
from kyc_router.schema import REQUIRED_FIELDS


PREDICTIONS = ("391378355", "AHMETI", "AGIM", "15-10-1973", "07-08-2025", "Shqiptare/Albanian", "M")


class FakeProcessor:
    def __init__(self):
        self.images = []
        self.decode_options = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.images.append(images.copy())
        return SimpleNamespace(pixel_values=torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    def decode(self, token_ids, **options):
        self.decode_options.append(options)
        return PREDICTIONS[int(token_ids[1])]


class FakeModel:
    def __init__(self):
        self.training = False
        self.calls = []
        self.config = SimpleNamespace(decoder=SimpleNamespace(use_cache=True))

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def generate(self, *, pixel_values, generation_config):
        assert not torch.is_grad_enabled()
        assert pixel_values.device.type == "cpu"
        assert pixel_values.dtype == torch.float32
        index = len(self.calls)
        self.calls.append((pixel_values.shape, generation_config))
        return torch.tensor([[2, index, 2]], dtype=torch.long)


def rectangles():
    return {name: {"x": 20 + index, "y": 15, "width": 12, "height": 8}
            for index, name in enumerate(REQUIRED_FIELDS)}


class OcrPipelineTests(unittest.TestCase):
    def setUp(self):
        self.processor = FakeProcessor()
        self.model = FakeModel()
        self.generation = SimpleNamespace(
            do_sample=False, num_beams=1, max_length=20, max_new_tokens=None, use_cache=False,
        )
        self.pipeline = OcrPipeline(
            self.processor, self.model, self.generation, device="cpu", canonical_size=(60, 40),
        )
        self.image = np.arange(40 * 60 * 3).astype(np.uint8).reshape(40, 60, 3)

    def test_seven_field_request(self):
        original = self.image.copy()
        result = self.pipeline.extract(self.image, rectangles())
        self.assertEqual(result["raw_fields"], dict(zip(REQUIRED_FIELDS, PREDICTIONS)))
        self.assertEqual(tuple(result["fields"]), REQUIRED_FIELDS)
        self.assertEqual(result["fields"]["surname"], "ahmeti")
        self.assertEqual(result["fields"]["date_of_birth"], "1973-10-15")
        self.assertTrue(result["validation"]["is_valid"])
        self.assertEqual(len(self.model.calls), 7)
        self.assertTrue(all(config is self.generation for _, config in self.model.calls))
        self.assertEqual(self.processor.decode_options, [
            {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        ] * 7)
        for index, field in enumerate(REQUIRED_FIELDS):
            details = result["field_details"][field]
            self.assertEqual(details["generated_token_ids"], [2, index, 2])
            bounds = details["crop_bounds"]
            expected = self.image[bounds["y1"]:bounds["y2"], bounds["x1"]:bounds["x2"]]
            np.testing.assert_array_equal(self.processor.images[index], expected)
            self.assertEqual(tuple(details["crop_shape"]), expected.shape)
        self.assertGreaterEqual(result["latency_ms"], result["inference_latency_ms"])
        self.assertGreaterEqual(result["inference_latency_ms"], 0)
        np.testing.assert_array_equal(self.image, original)

    def test_field_order_does_not_depend_on_caller_mapping_order(self):
        reversed_rectangles = dict(reversed(list(rectangles().items())))
        result = self.pipeline.extract(self.image, reversed_rectangles)
        self.assertEqual(tuple(result["raw_fields"]), REQUIRED_FIELDS)
        self.assertEqual(result["raw_fields"], dict(zip(REQUIRED_FIELDS, PREDICTIONS)))

    def test_rejects_noncanonical_pixels_before_model_work(self):
        cases = (
            (self.image.astype(np.float32), TypeError),
            (self.image[:, :, 0], ValueError),
            (self.image[:20], ValueError),
            (np.zeros((40, 60, 4), dtype=np.uint8), ValueError),
        )
        for image, error in cases:
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(error):
                self.pipeline.extract(image, rectangles())
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.processor.images, [])

    def test_rejects_incomplete_or_invalid_geometry_before_model_work(self):
        missing = rectangles()
        missing.pop("sex")
        extra = {**rectangles(), "extra": {"x": 1, "y": 1, "width": 1, "height": 1}}
        malformed = rectangles()
        malformed["sex"] = {"x": 1, "y": 1, "width": 0, "height": 1}
        empty = rectangles()
        empty["sex"] = {"x": 1000, "y": 1000, "width": 1, "height": 1}
        for shapes in (missing, extra, malformed, empty):
            with self.subTest(shapes=shapes), self.assertRaises((ValueError, TypeError)):
                self.pipeline.extract(self.image, shapes)
        self.assertEqual(self.model.calls, [])

    def test_training_mode_is_not_silently_accepted(self):
        self.model.training = True
        with self.assertRaises(RuntimeError):
            self.pipeline.extract(self.image, rectangles())
        self.assertEqual(self.model.calls, [])


class PipelineLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "checkpoints" / "selected"
        self.checkpoint.mkdir(parents=True)
        filenames = (
            "model.safetensors", "config.json", "generation_config.json",
            "processor_config.json", "tokenizer.json", "tokenizer_config.json",
            "selection_metadata.json",
        )
        hashes = {}
        for filename in filenames:
            contents = f"toy fixture: {filename}".encode()
            (self.checkpoint / filename).write_bytes(contents)
            hashes[filename] = hashlib.sha256(contents).hexdigest()
        self.manifest = {
            "pipeline_version": "ocr_v1",
            "document_type": "alb_id",
            "color": "RGB",
            "pixel_dtype": "uint8",
            "required_fields": list(REQUIRED_FIELDS),
            "precision": "float32",
            "field_batch_size": 1,
            "normalization_version": "v1",
            "validation_version": "alb_id_v1",
            "canonical_size": [1084, 680],
            "crop": {"padding_x": 0.1, "minimum_padding_x": 12, "padding_top": 0.1, "padding_bottom": 0.4},
            "generation": {
                "do_sample": False, "num_beams": 1, "max_length": 20,
                "max_new_tokens": None, "use_cache": False,
                "bos_token_id": 0, "decoder_start_token_id": 2, "pad_token_id": 1, "eos_token_id": 2,
            },
            "decode": {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
            "checkpoint_relative_path": "checkpoints/selected",
            "checkpoint_files_sha256": hashes,
        }
        self.manifest_path = self.root / "configs" / "ocr_pipeline_v1.json"
        self.manifest_path.parent.mkdir()
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def assert_fails_before_loading(self, error):
        with patch("kyc_router.ocr.TrOCRProcessor.from_pretrained") as processor_loader, \
                patch("kyc_router.ocr.VisionEncoderDecoderModel.from_pretrained") as model_loader, \
                patch("kyc_router.ocr.GenerationConfig.from_pretrained") as generation_loader:
            with self.assertRaises(error):
                load_pipeline(self.manifest_path)
            processor_loader.assert_not_called()
            model_loader.assert_not_called()
            generation_loader.assert_not_called()

    def test_missing_checkpoint_is_not_downloaded(self):
        (self.checkpoint / "model.safetensors").unlink()
        self.assert_fails_before_loading(FileNotFoundError)

    def test_corrupt_checkpoint_is_not_loaded(self):
        (self.checkpoint / "model.safetensors").write_bytes(b"tampered")
        self.assert_fails_before_loading(ValueError)

    def test_manifest_requires_complete_hashes(self):
        self.manifest["checkpoint_files_sha256"].pop("model.safetensors")
        self.write_manifest()
        self.assert_fails_before_loading(ValueError)

    def test_manifest_policy_changes_fail_before_loading(self):
        for field, changed in (
            ("precision", "float16"), ("canonical_size", [100, 100]), ("field_batch_size", 7),
            ("document_type", "other_id"), ("color", "BGR"), ("pixel_dtype", "float32"),
        ):
            original = self.manifest[field]
            self.manifest[field] = changed
            self.write_manifest()
            with self.subTest(field=field):
                self.assert_fails_before_loading(ValueError)
            self.manifest[field] = original
        self.manifest["generation"]["use_cache"] = True
        self.write_manifest()
        self.assert_fails_before_loading(ValueError)

    def test_hash_names_cannot_escape_checkpoint(self):
        self.manifest["checkpoint_files_sha256"]["../outside"] = "0" * 64
        self.write_manifest()
        self.assert_fails_before_loading(ValueError)

    def test_extra_generation_options_are_not_silently_applied(self):
        self.manifest["generation"]["num_return_sequences"] = 2
        self.write_manifest()
        self.assert_fails_before_loading(ValueError)

    def test_altered_special_tokens_are_rejected(self):
        self.manifest["generation"]["eos_token_id"] = 3
        self.write_manifest()
        self.assert_fails_before_loading(ValueError)

    def test_changed_decode_settings_are_rejected(self):
        for key, value in (("skip_special_tokens", False), ("clean_up_tokenization_spaces", True)):
            original = self.manifest["decode"][key]
            self.manifest["decode"][key] = value
            self.write_manifest()
            with self.subTest(option=key):
                self.assert_fails_before_loading(ValueError)
            self.manifest["decode"][key] = original

    def test_local_load_applies_frozen_settings(self):
        processor, model = FakeProcessor(), FakeModel()
        generation = SimpleNamespace()
        with patch("kyc_router.ocr.TrOCRProcessor.from_pretrained", return_value=processor) as processor_loader, \
                patch("kyc_router.ocr.VisionEncoderDecoderModel.from_pretrained", return_value=model) as model_loader, \
                patch("kyc_router.ocr.GenerationConfig.from_pretrained", return_value=generation) as generation_loader:
            pipeline = load_pipeline(self.manifest_path)
        processor_loader.assert_called_once_with(self.checkpoint, local_files_only=True)
        model_loader.assert_called_once_with(
            self.checkpoint, local_files_only=True, dtype=torch.float32, use_safetensors=True)
        generation_loader.assert_called_once_with(self.checkpoint, local_files_only=True)
        self.assertIs(pipeline.model, model)
        self.assertIs(pipeline.processor, processor)
        self.assertEqual(pipeline.canonical_size, (1084, 680))
        self.assertFalse(model.training)
        self.assertFalse(model.config.decoder.use_cache)
        self.assertIs(model.generation_config, generation)
        self.assertEqual(model.config.decoder_start_token_id, 2)
        for key, value in self.manifest["generation"].items():
            self.assertEqual(getattr(generation, key), value)


if __name__ == "__main__":
    unittest.main()
