"""The established sequential TrOCR path; no training or serving side effects."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel

from .geometry import check_rgb, extract_field_crops
from .normalization import normalize_document_values
from .schema import REQUIRED_FIELDS
from .validation import validate_document_values

_CROP_CONFIG = {
    "padding_x": 0.1, "minimum_padding_x": 12,
    "padding_top": 0.1, "padding_bottom": 0.4,
}
_GENERATION = {
    "do_sample": False, "num_beams": 1, "max_length": 20,
    "max_new_tokens": None, "use_cache": False,
    "bos_token_id": 0, "decoder_start_token_id": 2, "pad_token_id": 1, "eos_token_id": 2,
}


class OcrPipeline:
    """Keep model state between calls; extract one already-canonical document.

    Use load_pipeline() for the frozen model. Direct construction accepts the
    concrete processor/model objects, which also makes small tests possible.
    Calls are sequential, not a thread-safe serving scheduler.
    """

    def __init__(
        self, processor, model, generation_config, *, device="cpu",
        canonical_size=(1084, 680), crop_config=None,
    ):
        self.processor = processor
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only the established CPU and CUDA paths are supported.")
        self.model = model.to(self.device).eval()
        self.generation_config = generation_config
        self.canonical_size = tuple(canonical_size)
        self.crop_config = dict(_CROP_CONFIG if crop_config is None else crop_config)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def extract(self, canonical_rgb, rectangles: Mapping) -> dict:
        """Return normalized fields, raw evidence, checks, and completed timings.

        Input rectangles are in canonical pixel coordinates. No references,
        dataset identities, transcriptions, or output correction are accepted.
        Loading/rectification/scoring/writing are outside these request timers.
        """
        check_rgb(canonical_rgb, self.canonical_size)
        if self.model.training:
            raise RuntimeError("The OCR model is in training mode; call model.eval().")
        self._synchronize()
        start = time.perf_counter()
        crops = extract_field_crops(canonical_rgb, rectangles, self.crop_config)
        raw, details = {}, {}
        inference_ms = 0.0
        for field in REQUIRED_FIELDS:
            crop, bounds = crops[field]
            pixels = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
            self._synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    pixel_values=pixels, generation_config=self.generation_config)
            self._synchronize()
            inference_ms += (time.perf_counter() - inference_start) * 1000
            token_ids = generated[0].detach().cpu().tolist()
            raw[field] = self.processor.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            details[field] = {
                "crop_bounds": bounds, "crop_shape": list(crop.shape),
                "generated_token_ids": token_ids,
            }
        fields = normalize_document_values(raw)
        validation = validate_document_values(fields)
        self._synchronize()
        return {
            "fields": fields, "raw_fields": raw, "field_details": details,
            "validation": validation, "inference_latency_ms": inference_ms,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


def load_pipeline(manifest_path, *, device="cpu", artifact_root=None) -> OcrPipeline:
    """Verify and load the frozen local checkpoint; never download implicitly.

    Relative artifact paths resolve against artifact_root, defaulting to the
    repository root (parent of the manifest's configs/ directory), not cwd.
    Missing or altered weights fail before any Transformers loader is called.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pipeline_version") != "ocr_v1":
        raise ValueError("Unsupported OCR pipeline manifest version.")
    if tuple(manifest["required_fields"]) != REQUIRED_FIELDS:
        raise ValueError("Manifest field schema differs from the v1 contract.")
    if (manifest["document_type"] != "alb_id" or manifest["color"] != "RGB"
            or manifest["pixel_dtype"] != "uint8"
            or manifest["decode"] != {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}):
        raise ValueError("Manifest image/decode policy differs from the frozen v1 policy.")
    if (manifest["precision"] != "float32" or manifest["field_batch_size"] != 1
            or manifest["normalization_version"] != "v1"
            or manifest["validation_version"] != "alb_id_v1"
            or manifest["canonical_size"] != [1084, 680]
            or manifest["crop"] != _CROP_CONFIG):
        raise ValueError("Manifest preprocessing differs from the frozen v1 policy.")
    if manifest["generation"] != _GENERATION:
        raise ValueError("Manifest decoding differs from the frozen v1 policy.")
    root = Path(artifact_root).resolve() if artifact_root is not None else manifest_path.parent.parent
    checkpoint = root / manifest["checkpoint_relative_path"]
    hashes = manifest["checkpoint_files_sha256"]
    required = {"model.safetensors", "config.json", "generation_config.json",
                "processor_config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= set(hashes):
        raise ValueError("Manifest is missing required checkpoint hashes.")
    for filename, expected in hashes.items():
        # Hash entries name files, not arbitrary relative paths.
        if Path(filename).name != filename:
            raise ValueError(f"Invalid checkpoint filename: {filename!r}")
        path = checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing OCR artifact: {path}. See docs/guides/ocr-usage.md.")
        with path.open("rb") as artifact:
            digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    processor = TrOCRProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32, use_safetensors=True)
    generation = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    for key, value in manifest["generation"].items():
        setattr(generation, key, value)
    for key in ("decoder_start_token_id", "pad_token_id", "eos_token_id"):
        setattr(model.config, key, getattr(generation, key))
    model.config.decoder.use_cache = False
    model.generation_config = generation
    return OcrPipeline(
        processor, model, generation, device=device,
        canonical_size=manifest["canonical_size"], crop_config=manifest["crop"])
