"""Small explicit research operations shared by Phase 4 notebooks and runs.

MIDV loading/evaluation stay here, outside inference. This adapter refuses test
access before opening an image. No notebook is imported or executed by src/.
"""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoTokenizer, DonutImageProcessorPil, DonutProcessor, VisionEncoderDecoderModel

from kyc_router.geometry import rectangles_from_template
from kyc_router.normalization import normalize_document_values
from kyc_router.evaluation import score_document, summarize_scores
from kyc_router.schema import RAW_TO_CANONICAL, REQUIRED_FIELDS
from .donut_protocol import schema_tokens, serialize_fields, parse_sequence, TASK_TOKEN

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def new_run(group):
    path = ROOT / "outputs" / group / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path.mkdir(parents=True)
    return path


def setup(seed=2026):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


def environment():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor()),
        "logical_cpus": os.cpu_count(), "ram_gib": psutil.virtual_memory().total / 2**30,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_build": torch.version.cuda, "torch_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "transformers", "pillow", "numpy", "pandas", "opencv-python", "nbclient"]},
    }


def memory():
    return {"rss_mib": psutil.Process().memory_info().rss / 2**20,
            "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def load_observations(splits=("train", "validation"), identity_keys=None):
    if not set(splits) <= {"train", "validation"}:
        raise PermissionError("Test remains sealed throughout Phase 4.")
    manifest = pd.read_csv(ROOT / "splits/alb_id_v1.csv", dtype={"sample_id": "string"})
    assert manifest.identity_key.is_unique
    assert manifest.split.value_counts().to_dict() == {"train": 70, "validation": 15, "test": 15}
    chosen = manifest.loc[manifest.split.isin(splits)].sort_values("identity_key")
    if identity_keys is not None:
        if not set(identity_keys) <= set(chosen.identity_key):
            raise PermissionError("Requested identity is outside allowed splits.")
        chosen = chosen.loc[chosen.identity_key.isin(identity_keys)]
    root = ROOT / "data/raw"
    indices = {}
    allowed = {f"{sample}.jpg" for sample in chosen.sample_id}
    for source in ("templates", "photo"):
        meta = read_json(root / source / "annotations/alb_id.json")["_via_img_metadata"]
        indices[source] = {r["filename"]: r for r in meta.values() if r["filename"] in allowed}
        assert len(indices[source]) == len(chosen)
    def region(record, name):
        matches = [r for r in record["regions"] if r["region_attributes"].get("field_name") == name]
        assert len(matches) == 1
        return matches[0]
    observations = []
    for row in chosen.itertuples():
        name = f"{row.sample_id}.jpg"
        template = indices["templates"][name]
        photo = indices["photo"][name]
        images = {s: cv2.imread(str(root / s / "images/alb_id" / name)) for s in ("templates", "photo")}
        assert all(v is not None for v in images.values())
        truth_raw = {canonical: region(template, raw)["region_attributes"]["value"] for raw, canonical in RAW_TO_CANONICAL.items()}
        truth = normalize_document_values(truth_raw)
        # Photo VIA records provide document geometry, not seven transcriptions.
        # Both views use their identity-matched template's seven references.
        quad = region(photo, "doc_quad")["shape_attributes"]
        corners = np.asarray(list(zip(quad["all_points_x"], quad["all_points_y"])), dtype=np.float32)
        destination = np.asarray([[0, 0], [1083, 0], [1083, 679], [0, 679]], dtype=np.float32)
        rectified = cv2.warpPerspective(images["photo"], cv2.getPerspectiveTransform(corners, destination), (1084, 680))
        pixels = {"template": cv2.resize(images["templates"], (1084, 680), interpolation=cv2.INTER_AREA), "photo": rectified}
        h, w = images["templates"].shape[:2]
        rectangles = rectangles_from_template(template, w, h)
        for source, bgr in pixels.items():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            observations.append({"identity_key": row.identity_key, "split": row.split, "source": source,
                                 "image": rgb, "truth_raw": truth_raw, "truth": truth,
                                 "rectangles": rectangles, "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    return observations


def load_candidate(policy, *, device="cpu", extend=True):
    checkpoint = ROOT / policy["parent_checkpoint"]
    manifest = read_json(checkpoint / "artifact_manifest.json")
    assert manifest["revision"] == policy["parent_revision"]
    for name, digest in manifest["files_sha256"].items():
        assert sha(checkpoint / name) == digest, name
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ip = DonutImageProcessorPil.from_pretrained(
        checkpoint, local_files_only=True, size=policy["processor"]["size"],
        do_align_long_axis=policy["processor"]["do_align_long_axis"],
    )
    processor = DonutProcessor(image_processor=ip, tokenizer=tokenizer)
    model, info = VisionEncoderDecoderModel.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True,
    )
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    if extend:
        added = tokenizer.add_special_tokens({"additional_special_tokens": schema_tokens()})
        assert added == 15
        model.decoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.decoder.vocab_size = len(tokenizer)
    model.to(device).eval()
    return processor, model


def target_ids(processor, fields, limit=192):
    sequence = serialize_fields(fields)
    ids = processor.tokenizer(sequence, add_special_tokens=False)["input_ids"]
    assert len(ids) <= limit, f"Target truncation forbidden: {len(ids)} > {limit}"
    assert ids[0] == processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
    assert ids[-1] == processor.tokenizer.eos_token_id
    assert parse_sequence(sequence)["fields"] == fields
    decoded = processor.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert normalize_document_values(parse_sequence(decoded)["fields"]) == fields
    return ids


def teacher_forcing(ids, device):
    # Prefix at position t predicts the following target at t+1. No second
    # implicit shift is applied: MBartForCausalLM here compares aligned logits.
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    return tokens[:, :-1], tokens[:, 1:]


def generation_kwargs(processor, policy):
    tokenizer = processor.tokenizer
    return {**policy["generation"], "decoder_start_token_id": tokenizer.convert_tokens_to_ids(TASK_TOKEN),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id}


def research_predict(processor, model, image, policy, *, encoder_features=None):
    device = next(model.decoder.parameters()).device
    prompt = torch.tensor([[processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)]], device=device)
    with torch.inference_mode():
        if encoder_features is None:
            pixels = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated = model.generate(pixel_values=pixels, decoder_input_ids=prompt,
                                       **generation_kwargs(processor, policy))
        else:
            from transformers.modeling_outputs import BaseModelOutput
            assert next(model.encoder.parameters()).device == device, "Co-locate encoder/decoder for parent generation."
            generated = model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_features.to(device)),
                decoder_input_ids=prompt, **generation_kwargs(processor, policy))
    ids = generated[0].cpu().tolist()
    sequence = processor.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    parsed = parse_sequence(sequence)
    return {"token_ids": ids, "raw_sequence": sequence, "parse": parsed,
            "raw_fields": parsed["fields"], "eos_emitted": processor.tokenizer.eos_token_id in ids[1:]}


def evaluate_predictions(observations, predictions):
    fields, docs = [], []
    assert len(observations) == len(predictions)
    for obs, pred in zip(observations, predictions):
        field_rows, doc = score_document(obs["truth"], pred["raw_fields"])
        key = {"identity_key": obs["identity_key"], "source": obs["source"]}
        fields.extend([{**key, **row} for row in field_rows])
        docs.append({**key, **doc})
    fs, ds = pd.DataFrame(fields), pd.DataFrame(docs)
    per_field, summary = summarize_scores(fs, ds)
    return fs, ds, per_field, summary
