"""Whole-image inference and local artifact loading without dataset/GPU access."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kyc_router.donut import DonutPipeline, load_donut_pipeline
from kyc_router.evaluation import score_document
from kyc_router.schema import REQUIRED_FIELDS
from kyc_router.serialization import TASK_TOKEN, schema_tokens, serialize_fields
from test_serialization import TRUTH


def generation():
    return {"do_sample": False, "num_beams": 1, "max_new_tokens": 192,
            "max_length": None, "use_cache": True, "forced_eos_token_id": None,
            "decoder_start_token_id": 100, "bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1}


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1

    def __init__(self, sequence=None):
        self.symbols = dict(zip(schema_tokens(), range(100, 115)))
        self.sequence = serialize_fields(TRUTH) if sequence is None else sequence
        self.decode_options = []

    def __call__(self, text, *, add_special_tokens, return_tensors=None):
        ids = [self.symbols[text]]
        return {"input_ids": torch.tensor([ids]) if return_tensors else ids}

    def convert_tokens_to_ids(self, text):
        return self.symbols[text]

    def __len__(self):
        return 200

    def decode(self, ids, **kwargs):
        self.decode_options.append(kwargs)
        return self.sequence


class Processor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer()
        self.images = []

    def __call__(self, *, images, return_tensors):
        self.images.append(images.copy())
        return SimpleNamespace(pixel_values=torch.zeros((1, 3, 4, 6)))


class Model:
    def __init__(self):
        self.training = False
        self.calls = []
        self.decoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(num_embeddings=200))

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append(kwargs)
        return torch.tensor([[100, 120, 2]])


class DonutRequestTests(unittest.TestCase):
    def setUp(self):
        self.processor, self.model = Processor(), Model()
        self.pipeline = DonutPipeline(self.processor, self.model, generation(), canonical_size=(6, 4))
        self.image = np.zeros((4, 6, 3), dtype=np.uint8)

    def test_whole_image_single_call_and_shared_rules(self):
        result = self.pipeline.extract(self.image)
        self.assertEqual(result["fields"], TRUTH)
        self.assertEqual(tuple(result["fields"]), REQUIRED_FIELDS)
        self.assertTrue(result["validation"]["is_valid"])
        self.assertEqual(len(self.model.calls), 1)
        np.testing.assert_array_equal(self.processor.images[0], self.image)
        self.assertEqual(self.processor.images[0].shape, (4, 6, 3))
        self.assertTrue(result["eos_emitted"])
        self.assertGreaterEqual(result["latency_ms"], result["inference_latency_ms"])
        self.assertIn("decoder_input_ids", self.model.calls[0])
        self.assertIsNone(self.model.calls[0]["forced_eos_token_id"])
        self.assertEqual(self.processor.tokenizer.decode_options, [{"skip_special_tokens": False, "clean_up_tokenization_spaces": False}])

    def test_structural_failure_preserves_raw_text_and_scores_missing(self):
        raw = "<s_alb_id><s_surname>broken"
        self.processor.tokenizer.sequence = raw
        result = self.pipeline.extract(self.image)
        self.assertEqual(result["raw_sequence"], raw)
        self.assertIsNone(result["raw_fields"])
        self.assertTrue(all(v is None for v in result["fields"].values()))
        self.assertFalse(result["validation"]["schema_is_valid"])
        self.assertFalse(result["validation"]["is_valid"])
        rows, doc = score_document(TRUTH, result["raw_fields"])
        self.assertTrue(doc["structural_failure"])
        self.assertEqual(sum(r["normalized_exact_match"] for r in rows), 0)

    def test_partial_and_extra_keys_are_retained_for_scoring(self):
        self.processor.tokenizer.sequence = serialize_fields(TRUTH).replace("<s_sex>M</s_sex>", "<s_extra>x</s_extra>")
        result = self.pipeline.extract(self.image)
        self.assertIn("extra", result["raw_fields"])
        self.assertIsNone(result["fields"]["sex"])
        self.assertFalse(result["validation"]["is_valid"])
        self.assertEqual(result["validation"]["missing_fields"], ("sex",))
        self.assertEqual(result["validation"]["unexpected_fields"], ("extra",))

    def test_noncanonical_input_fails_before_model_work(self):
        for image in [self.image[:2], self.image[:, :, 0], self.image.astype(np.float32)]:
            with self.assertRaises((ValueError, TypeError)):
                self.pipeline.extract(image)
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.processor.images, [])

    def test_training_mode_fails_before_model_work(self):
        self.model.training = True
        with self.assertRaises(RuntimeError):
            self.pipeline.extract(self.image)
        self.assertEqual(self.model.calls, [])


class DonutLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoints/test"
        self.checkpoint.mkdir(parents=True)
        names = ["model.safetensors", "config.json", "generation_config.json", "processor_config.json",
                 "tokenizer.json", "tokenizer_config.json", "selection_metadata.json"]
        hashes = {}
        for name in names:
            data = f"fixture {name}".encode()
            (self.checkpoint / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        self.manifest = {
            "pipeline_version": "donut_v1", "document_type": "alb_id", "required_fields": list(REQUIRED_FIELDS),
            "canonical_size": [1084, 680], "color": "RGB", "pixel_dtype": "uint8", "precision": "float32",
            "normalization_version": "v1", "validation_version": "alb_id_v1", "serialization_version": "alb_id_tags_v1",
            "batch_size": 1, "generation": generation(), "checkpoint_relative_path": "checkpoints/test",
            "checkpoint_files_sha256": hashes, "schema_token_ids": Tokenizer().symbols,
            "processor": {"backend": "pil", "size": {"height": 640, "width": 1024}, "do_align_long_axis": False},
            "attention_implementation": "eager",
            "decode": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False,
                       "strip_trailing_padding_ids_only": True},
        }
        self.path = self.root / "configs/donut_pipeline_v1.json"
        self.path.parent.mkdir()
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def rejected_before_tokenizer(self, error):
        with patch("kyc_router.donut.AutoTokenizer.from_pretrained") as tokenizer:
            with self.assertRaises(error):
                load_donut_pipeline(self.path)
            tokenizer.assert_not_called()

    def test_missing_and_corrupt_weights_never_download(self):
        (self.checkpoint / "model.safetensors").unlink()
        self.rejected_before_tokenizer(FileNotFoundError)
        (self.checkpoint / "model.safetensors").write_bytes(b"tampered")
        self.rejected_before_tokenizer(ValueError)

    def test_unsupported_contract_and_decode_fail_early(self):
        self.manifest["serialization_version"] = "unknown"
        self.save()
        self.rejected_before_tokenizer(ValueError)
        self.manifest["serialization_version"] = "alb_id_tags_v1"
        self.manifest["generation"]["num_beams"] = 2
        self.save()
        self.rejected_before_tokenizer(ValueError)

    def test_hash_names_and_checkpoint_path_cannot_escape_root(self):
        self.manifest["checkpoint_files_sha256"]["../escape"] = "0" * 64
        self.save()
        self.rejected_before_tokenizer(ValueError)
        self.manifest["checkpoint_relative_path"] = "../outside"
        self.save()
        self.rejected_before_tokenizer(ValueError)

    def test_changed_decode_policy_is_not_silently_ignored(self):
        self.manifest["decode"]["skip_special_tokens"] = True
        self.save()
        self.rejected_before_tokenizer(ValueError)

    def test_required_hashes_are_not_optional(self):
        self.manifest["checkpoint_files_sha256"].pop("tokenizer.json")
        self.save()
        self.rejected_before_tokenizer(ValueError)

    def test_verified_load_is_offline_and_applies_policy(self):
        tokenizer, model = Tokenizer(), Model()
        ip = SimpleNamespace(size=SimpleNamespace(height=640, width=1024), do_align_long_axis=False)
        with patch("kyc_router.donut.AutoTokenizer.from_pretrained", return_value=tokenizer) as tl, \
             patch("kyc_router.donut.DonutImageProcessorPil.from_pretrained", return_value=ip) as pl, \
             patch("kyc_router.donut.VisionEncoderDecoderModel.from_pretrained", return_value=model) as ml, \
             patch("kyc_router.donut.DonutProcessor", return_value=Processor(tokenizer)):
            pipeline = load_donut_pipeline(self.path)
        tl.assert_called_once_with(self.checkpoint, local_files_only=True)
        pl.assert_called_once_with(self.checkpoint, local_files_only=True)
        self.assertTrue(ml.call_args.kwargs["local_files_only"])
        self.assertTrue(ml.call_args.kwargs["use_safetensors"])
        self.assertEqual(pipeline.generation, generation())
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
