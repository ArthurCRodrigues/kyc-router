"""Whole-card Donut inference with strict parsing and the shared field rules.

No dataset loading, identity/reference input, training features, fallback, or
serving infrastructure. Use load_donut_pipeline for verified local artifacts.
"""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, DonutImageProcessorPil, DonutProcessor, VisionEncoderDecoderModel

from .geometry import check_rgb
from .normalization import normalize_document_values
from .schema import REQUIRED_FIELDS
from .serialization import TASK_TOKEN, parse_sequence, schema_tokens
from .validation import validate_document_values


class DonutPipeline:
    """One loaded model, one canonical whole image per call; sequential use."""

    def __init__(self, processor, model, generation, *, device="cpu", canonical_size=(1084, 680)):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA devices are supported.")
        self.processor = processor
        self.model = model.to(self.device).eval()
        self.generation = dict(generation)
        self.canonical_size = tuple(canonical_size)
        self.prompt_ids = processor.tokenizer(TASK_TOKEN, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        if self.prompt_ids.shape != (1, 1):
            raise ValueError("The trained task prefix must be a single registered token.")

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def extract(self, canonical_rgb):
        """Return seven normalized fields plus unmodified structural evidence.

        Parsed raw_fields retains missing/extra keys; None records a whole
        structural failure for scoring. fields always contains seven canonical
        keys, using None when the corresponding prediction is unavailable.
        Model-specific preprocessing through validation is inside latency_ms.
        """
        check_rgb(canonical_rgb, self.canonical_size)
        if self.model.training:
            raise RuntimeError("The Donut model is in training mode; call model.eval().")
        self._synchronize()
        start = time.perf_counter()
        pixels = self.processor(images=canonical_rgb, return_tensors="pt").pixel_values.to(self.device)
        self._synchronize()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(pixel_values=pixels, decoder_input_ids=self.prompt_ids,
                                            **self.generation)
        self._synchronize()
        inference_ms = (time.perf_counter() - inference_start) * 1000
        ids = generated[0].detach().cpu().tolist()
        decode_ids = list(ids)
        while decode_ids and decode_ids[-1] == self.generation["pad_token_id"]:
            decode_ids.pop()
        sequence = self.processor.tokenizer.decode(
            decode_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        parsed = parse_sequence(sequence)
        raw = parsed["fields"]
        fields = normalize_document_values({f: None if raw is None else raw.get(f) for f in REQUIRED_FIELDS})
        validation = validate_document_values(fields)
        missing = sorted(set(REQUIRED_FIELDS) - set(raw or {}))
        extra = sorted(set(raw or {}) - set(REQUIRED_FIELDS))
        schema_valid = not parsed["structural_failure"] and not missing and not extra
        validation.update(schema_is_valid=schema_valid, missing_fields=tuple(missing), unexpected_fields=tuple(extra))
        validation["is_valid"] = validation["is_valid"] and schema_valid
        eos = self.generation["eos_token_id"] in ids[self.prompt_ids.shape[1]:]
        self._synchronize()
        return {
            "fields": fields, "raw_fields": raw, "raw_sequence": sequence,
            "generated_token_ids": ids, "parse": parsed, "validation": validation,
            "eos_emitted": eos,
            "length_limit_reached_without_eos": not eos and len(ids) - self.prompt_ids.shape[1] >= self.generation["max_new_tokens"],
            "inference_latency_ms": inference_ms, "latency_ms": (time.perf_counter() - start) * 1000,
        }


def load_donut_pipeline(manifest_path, *, device="cpu", artifact_root=None):
    """Verify a frozen local model before loading; never download implicitly."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get("pipeline_version") != "donut_v1"
            or manifest.get("document_type") != "alb_id"
            or tuple(manifest.get("required_fields", ())) != REQUIRED_FIELDS
            or manifest.get("canonical_size") != [1084, 680]
            or manifest.get("color") != "RGB" or manifest.get("pixel_dtype") != "uint8"
            or manifest.get("precision") != "float32"
            or manifest.get("normalization_version") != "v1"
            or manifest.get("validation_version") != "alb_id_v1"
            or manifest.get("serialization_version") != "alb_id_tags_v1"
            or manifest.get("batch_size") != 1
            or manifest.get("attention_implementation") != "eager"
            or manifest.get("decode") != {"skip_special_tokens": False, "clean_up_tokenization_spaces": False,
                                          "strip_trailing_padding_ids_only": True}):
        raise ValueError("Manifest differs from the supported Donut v1 contract.")
    generation = manifest["generation"]
    expected_keys = {"do_sample", "num_beams", "max_new_tokens", "max_length", "use_cache", "forced_eos_token_id",
                     "decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id"}
    if (set(generation) != expected_keys or generation["do_sample"] is not False
            or generation["num_beams"] != 1 or generation["use_cache"] is not True
            or generation["max_length"] is not None or generation["forced_eos_token_id"] is not None
            or not isinstance(generation["max_new_tokens"], int) or generation["max_new_tokens"] <= 0):
        raise ValueError("Unsupported Donut decoding policy.")
    root = Path(artifact_root).resolve() if artifact_root is not None else path.parent.parent
    relative = Path(manifest["checkpoint_relative_path"])
    checkpoint = (root / relative).resolve()
    if relative.is_absolute() or not checkpoint.is_relative_to(root):
        raise ValueError("Checkpoint path must remain inside artifact_root.")
    hashes = manifest["checkpoint_files_sha256"]
    required = {"model.safetensors", "config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "selection_metadata.json"}
    if not required <= set(hashes):
        raise ValueError("Manifest lacks required Donut artifact hashes.")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Artifact hash names must be plain filenames.")
        artifact = checkpoint / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing Donut artifact: {artifact}. See docs/guides/donut-usage.md.")
        with artifact.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Donut artifact SHA-256 mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ip = DonutImageProcessorPil.from_pretrained(checkpoint, local_files_only=True)
    if (manifest["processor"]["backend"] != "pil"
            or {"height": ip.size.height, "width": ip.size.width} != manifest["processor"]["size"]
            or ip.do_align_long_axis != manifest["processor"]["do_align_long_axis"]):
        raise ValueError("Loaded image processor differs from the manifest.")
    for symbol, expected in manifest["schema_token_ids"].items():
        ids = tokenizer(symbol, add_special_tokens=False)["input_ids"]
        if ids != [expected]:
            raise ValueError(f"Tokenizer schema symbol changed: {symbol}")
    if set(manifest["schema_token_ids"]) != set(schema_tokens()):
        raise ValueError("Manifest must identify all task/field token IDs.")
    if (generation["decoder_start_token_id"] != tokenizer.convert_tokens_to_ids(TASK_TOKEN)
            or generation["bos_token_id"] != tokenizer.bos_token_id
            or generation["eos_token_id"] != tokenizer.eos_token_id
            or generation["pad_token_id"] != tokenizer.pad_token_id):
        raise ValueError("Generation IDs disagree with the saved tokenizer.")
    model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32,
        use_safetensors=True, attn_implementation=manifest["attention_implementation"])
    if len(tokenizer) != model.decoder.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer vocabulary and decoder embedding sizes disagree.")
    processor = DonutProcessor(image_processor=ip, tokenizer=tokenizer)
    return DonutPipeline(processor, model, generation, device=device, canonical_size=manifest["canonical_size"])
