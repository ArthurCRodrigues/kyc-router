"""Freeze the validation-selected local model and its exact artifacts/policy."""

import argparse

from transformers import AutoTokenizer

from kyc_router.schema import REQUIRED_FIELDS
from kyc_router.serialization import schema_tokens
from .donut_workshop import ROOT, generation_kwargs, read_json, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_run")
    args = parser.parse_args()
    run = ROOT / args.training_run
    completed = read_json(run / "completion.json")
    assert completed["status"] == "completed" and completed["frozen_encoder_verified"]
    selected = completed["selected"]
    checkpoint = ROOT / selected["checkpoint"]
    assert read_json(checkpoint / "selection_metadata.json") == selected
    policy = read_json(run / "dataset_policy.json")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    processor = type("TokenizerContainer", (), {"tokenizer": tokenizer})()
    manifest = {
        "pipeline_version": "donut_v1", "status": "frozen validation-selected Phase 4 path",
        "document_type": "alb_id", "required_fields": list(REQUIRED_FIELDS),
        "canonical_size": [1084, 680], "color": "RGB", "pixel_dtype": "uint8",
        "precision": "float32", "batch_size": 1,
        "processor": policy["processor"], "attention_implementation": "eager",
        "generation": generation_kwargs(processor, policy),
        "decode": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False,
                   "strip_trailing_padding_ids_only": True},
        "serialization_version": policy["serialization"]["version"],
        "serialization_policy": policy["serialization"],
        "schema_token_ids": {symbol: tokenizer.convert_tokens_to_ids(symbol) for symbol in schema_tokens()},
        "normalization_version": "v1", "validation_version": "alb_id_v1",
        "parent_model_id": policy["parent_model_id"], "parent_revision": policy["parent_revision"],
        "checkpoint_relative_path": selected["checkpoint"],
        "checkpoint_files_sha256": {p.name: sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
        "selection": selected, "training_evidence": str(run.relative_to(ROOT)),
        "training_configuration": "configs/donut_finetuning_v2.json",
        "training_configuration_sha256": selected["training_config_sha256"],
        "dataset_configuration": "configs/donut_dataset_v1.json",
        "dataset_configuration_sha256": selected["dataset_policy_sha256"],
        "split_manifest": "splits/alb_id_v1.csv", "split_manifest_sha256": sha(ROOT / "splits/alb_id_v1.csv"),
        "adaptation": "frozen pretrained encoder, all decoder parameters trainable; Adafactor float32",
        "training_feature_cache_in_inference": False,
        "latency_protocol": "configs/ocr_donut_comparison_v1.json",
        "test_evaluated": False,
    }
    path = ROOT / "configs/donut_pipeline_v1.json"
    if path.exists():
        assert read_json(path) == manifest, "Do not overwrite a different frozen model."
    write_json(path, manifest)
    print("FROZEN", path.relative_to(ROOT), "epoch", selected["epoch"])


if __name__ == "__main__":
    main()
