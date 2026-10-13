"""Validate notebook-to-source extraction without fitting or opening test data.

Run from an installed checkout:
    python -m experiments.verify_ocr_source --device cuda:0

Notebook execution is confined to the reference side of this migration check.
The actual package never imports or executes notebooks.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch

from kyc_router.evaluation import score_document, summarize_scores
from kyc_router.geometry import crop_field, rectangles_from_template
from kyc_router.ocr import load_pipeline
from kyc_router.schema import REQUIRED_FIELDS
from kyc_router.validation import validate_document_values

from .ocr_resource_profile import load_notebook_contract

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/ocr_pipeline_v1.json"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify(device):
    # Explicit experiment controls, not package-wide import side effects.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(6)
    cv2.setNumThreads(1)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("split_manifest", "split_manifest_sha256"),
        ("geometry_configuration", "geometry_configuration_sha256"),
        ("training_configuration", "training_configuration_sha256"),
    ):
        if sha(ROOT / manifest[path_key]) != manifest[hash_key]:
            raise ValueError(f"Frozen provenance mismatch: {path_key}")
    with contextlib.redirect_stdout(io.StringIO()):
        reference, notebook_hashes = load_notebook_contract()
    identities = reference["split_manifest"].query("split == 'validation'").identity_key.tolist()
    if len(identities) != 15 or len(set(identities)) != 15:
        raise ValueError("Expected 15 distinct validation identities.")
    evidence = ROOT / manifest["validation_evidence"]
    expected_fields = {(row["identity_key"], row["source"], row["field"]): row
                       for row in json_rows(evidence / "selected_reload_fields.jsonl")}
    expected_documents = {(row["identity_key"], row["source"]): row
                          for row in json_rows(evidence / "selected_reload_documents.jsonl")}
    pipeline = load_pipeline(MANIFEST, device=device)
    observations, geometry_checks = [], 0
    for identity in identities:
        assets = reference["load_alb_id_assets"](
            identity, manifest_index=reference["manifest_by_identity"],
            dataset_root=reference["dataset_root"],
            template_record_index=reference["template_records_by_filename"],
            photo_record_index=reference["photo_records_by_filename"])
        if assets["split"] != "validation":
            raise PermissionError("Migration checks may use validation only.")
        for source in ("template", "photo"):
            obs = reference["make_canonical_observation"](assets, source, 1084, 680)
            rectangles = rectangles_from_template(
                obs["template_record"], obs["template_source_width"], obs["template_source_height"])
            old_crops = reference["extract_field_crops"](
                obs, reference["raw_to_canonical_field"], reference["field_crop_config"])
            for field in REQUIRED_FIELDS:
                crop, bounds = crop_field(obs["canonical_rgb"], rectangles[field])
                if rectangles[field] != old_crops[field]["canonical_shape"] or bounds != old_crops[field]["crop_bounds"]:
                    raise AssertionError("Geometry changed during extraction.")
                np.testing.assert_array_equal(crop, old_crops[field]["crop_rgb"])
                geometry_checks += 1
            truth = {field: old_crops[field]["ground_truth"] for field in REQUIRED_FIELDS}
            observations.append((identity, source, obs["canonical_rgb"], rectangles, truth))
    scored_fields, scored_documents, predictions = [], [], []
    for identity, source, pixels, rectangles, truth in observations:
        result = pipeline.extract(pixels, rectangles)
        field_scores, document_score = score_document(truth, result["raw_fields"])
        old_scores, old_document = reference["score_document"](
            truth, result["raw_fields"], required_fields=REQUIRED_FIELDS)
        if field_scores != old_scores or document_score != old_document:
            raise AssertionError("Shared scoring changed during extraction.")
        expected_doc = expected_documents[(identity, source)]
        for key, value in document_score.items():
            if value != expected_doc[key]:
                raise AssertionError(f"Document scoring mismatch: {key}")
        old_validation = reference["validate_document_values"](
            result["fields"], required_fields=REQUIRED_FIELDS,
            validation_config=reference["alb_id_validation_config"])
        if result["validation"] != old_validation:
            raise AssertionError("Validator changed during extraction.")
        for row in field_scores:
            field = row["field"]
            expected = expected_fields[(identity, source, field)]
            for key, value in row.items():
                if value != expected[key]:
                    raise AssertionError(f"Field prediction/scoring mismatch: {identity}/{source}/{field}/{key}")
            if result["field_details"][field]["generated_token_ids"] != expected["generated_token_ids"]:
                raise AssertionError("Generated token sequence changed.")
            scored_fields.append({"identity_key": identity, "source": source, **row})
        scored_documents.append({"identity_key": identity, "source": source, **document_score})
        predictions.append({"identity_key": identity, "source": source, "split": "validation", **result})
    per_field, summary = summarize_scores(pd.DataFrame(scored_fields), pd.DataFrame(scored_documents))
    expected_summary = pd.read_csv(evidence / "selected_reload_summary.csv")
    pd.testing.assert_frame_equal(summary, expected_summary, check_dtype=False, atol=1e-12, rtol=1e-12)
    # Same warm-up/repetition policy as the retained fine-tuning comparison.
    for source in ("template", "photo"):
        for row in [obs for obs in observations if obs[1] == source][:3]:
            pipeline.extract(row[2], row[3])
    rng, timings = np.random.default_rng(2026), []
    for repetition in range(3):
        for index in rng.permutation(len(observations)):
            identity, source, pixels, rectangles, _ = observations[int(index)]
            result = pipeline.extract(pixels, rectangles)
            expected = next(p for p in predictions if p["identity_key"] == identity and p["source"] == source)
            if result["raw_fields"] != expected["raw_fields"] or result["field_details"] != expected["field_details"]:
                raise AssertionError("A timed repeat changed predictions or token sequences.")
            timings.append({"identity_key": identity, "source": source, "repetition": repetition,
                            "inference_latency_ms": result["inference_latency_ms"], "latency_ms": result["latency_ms"]})
    timing_table = pd.DataFrame(timings)
    latency = timing_table.groupby("source", as_index=False).agg(
        requests=("latency_ms", "size"), inference_mean_ms=("inference_latency_ms", "mean"),
        inference_median_ms=("inference_latency_ms", "median"),
        end_to_end_mean_ms=("latency_ms", "mean"), end_to_end_median_ms=("latency_ms", "median"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "outputs/phase3_closure" / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "predictions.json", predictions)
    per_field.to_csv(output / "per_field.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    timing_table.to_csv(output / "latency_measurements.csv", index=False)
    latency.to_csv(output / "latency_summary.csv", index=False)
    report = {
        "run_id": run_id, "pipeline_manifest": str(MANIFEST.relative_to(ROOT)),
        "pipeline_manifest_sha256": sha(MANIFEST), "validation_identities": identities,
        "validation_observations": len(observations), "geometry_crop_parity_checks": geometry_checks,
        "raw_prediction_and_token_parity_checks": len(scored_fields),
        "normalized_scoring_and_validation_parity": True,
        "timing_repeats_with_exact_prediction_and_token_agreement": len(timings),
        "summary": summary.to_dict(orient="records"), "latency": latency.to_dict(orient="records"),
        "test_evaluated": False, "training_performed": False,
        "historical_notebook_code_sha256": notebook_hashes,
        "source_files_sha256": {str(p.relative_to(ROOT)): sha(p)
                                for p in sorted((ROOT / "src/kyc_router").glob("*.py"))},
        "environment": {
            "python": platform.python_version(), "os": platform.platform(), "device": str(pipeline.device),
            "gpu": torch.cuda.get_device_name(pipeline.device) if pipeline.device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "opencv_threads": cv2.getNumThreads(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("kyc-router", "torch", "torchvision", "transformers", "numpy", "pandas", "opencv-python")},
        },
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "source_state": "dirty checkout; content hashes above identify actual implementation",
    }
    write_json(output / "verification.json", report)
    write_json(output / "artifact_hashes.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Explicit device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    print(verify(args.device))
