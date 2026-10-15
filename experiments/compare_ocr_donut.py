"""Verify the extracted Donut path and compare fixed pipelines on validation.

    .venv/bin/python -m experiments.compare_ocr_donut

Canonical images are identical for both models. Models are individually resident;
all reference scoring, comparisons, logging, and file writes are outside request
timers. There is no training-feature cache in inference or measured timing.
"""

import gc
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from kyc_router.donut import load_donut_pipeline
from kyc_router.ocr import load_pipeline
from kyc_router.serialization import parse_sequence as source_parse, serialize_fields as source_serialize
from .donut_protocol import parse_sequence as research_parse, serialize_fields as research_serialize
from .donut_workshop import (
    ROOT, environment, evaluate_predictions, load_observations, memory, new_run,
    read_json, research_predict, setup, sha, write_json,
)


def main():
    policy_path = ROOT / "configs/ocr_donut_comparison_v1.json"
    policy = read_json(policy_path)
    setup(policy["seed"])
    run = new_run("phase4_comparison")
    write_json(run / "policy.json", policy)
    write_json(run / "environment.json", environment())
    source_paths = ["src/kyc_router/donut.py", "src/kyc_router/serialization.py", "src/kyc_router/ocr.py",
                    "src/kyc_router/normalization.py", "src/kyc_router/validation.py", "src/kyc_router/evaluation.py",
                    "experiments/compare_ocr_donut.py", "experiments/donut_workshop.py", "experiments/donut_protocol.py",
                    "configs/donut_pipeline_v1.json", "configs/ocr_pipeline_v1.json", "configs/ocr_donut_comparison_v1.json"]
    write_json(run / "source_hashes.json", {p: sha(ROOT / p) for p in source_paths})
    for name in source_paths:
        destination = run / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    observations = load_observations(splits=("validation",))
    assert len(observations) == 30
    manifest = read_json(ROOT / "configs/donut_pipeline_v1.json")
    training = ROOT / manifest["training_evidence"]
    selected = read_json(training / f"validation_epoch_{manifest['selection']['epoch']:02d}.json")
    selected_index = {(p["identity_key"], p["source"]): p for p in selected}
    pipeline = load_donut_pipeline(ROOT / "configs/donut_pipeline_v1.json", device="cuda:0")
    dataset_policy = read_json(training / "dataset_policy.json")
    verified = []
    print("RUN", run.relative_to(ROOT), flush=True)
    for i, obs in enumerate(observations):
        actual = pipeline.extract(obs["image"])
        reference = research_predict(pipeline.processor, pipeline.model, obs["image"], dataset_policy)
        retained = selected_index[obs["identity_key"], obs["source"]]
        assert actual["generated_token_ids"] == reference["token_ids"] == retained["token_ids"]
        assert actual["raw_sequence"] == reference["raw_sequence"] == retained["raw_sequence"]
        assert actual["raw_fields"] == reference["raw_fields"] == retained["raw_fields"]
        assert actual["parse"] == research_parse(actual["raw_sequence"])
        assert source_serialize(obs["truth"]) == research_serialize(obs["truth"])
        verified.append({"identity_key": obs["identity_key"], "source": obs["source"],
                         "pixel_sha256": obs["pixel_sha256"], "result": actual})
        if (i + 1) % 10 == 0:
            print("PARITY", i + 1, "/ 30", flush=True)
    write_json(run / "donut_verification.json", verified)
    del pipeline, actual, reference
    gc.collect()
    torch.cuda.empty_cache()

    rng = np.random.default_rng(policy["seed"])
    orders = [rng.permutation(len(observations)).tolist() for _ in range(3)]
    occurrences = {"ocr": 0, "donut": 0}
    baselines = {}
    timings, blocks = [], []
    for block_index, kind in enumerate(policy["block_order"]):
        pipeline = (load_pipeline(ROOT / "configs/ocr_pipeline_v1.json", device="cuda:0") if kind == "ocr"
                    else load_donut_pipeline(ROOT / "configs/donut_pipeline_v1.json", device="cuda:0"))
        def extract(obs):
            return pipeline.extract(obs["image"], obs["rectangles"]) if kind == "ocr" else pipeline.extract(obs["image"])
        for source in ("template", "photo"):
            warmup = [o for o in observations if o["source"] == source][:policy["warmup_documents_per_source_per_block"]]
            for obs in warmup:
                extract(obs)
        torch.cuda.reset_peak_memory_stats()
        repetition = occurrences[kind]
        order = orders[repetition]
        predictions = [None] * len(observations)
        print("BLOCK", block_index + 1, kind, "repetition", repetition + 1, flush=True)
        for index in order:
            obs = observations[index]
            result = extract(obs)
            key = (kind, obs["identity_key"], obs["source"])
            if kind == "ocr":
                ids = {f: d["generated_token_ids"] for f, d in result["field_details"].items()}
            else:
                ids = result["generated_token_ids"]
                assert ids == selected_index[obs["identity_key"], obs["source"]]["token_ids"]
            if key in baselines:
                assert ids == baselines[key]["tokens"]
                assert result["fields"] == baselines[key]["fields"]
                assert result["validation"] == baselines[key]["validation"]
            else:
                baselines[key] = {"tokens": ids, "fields": result["fields"], "validation": result["validation"]}
            predictions[index] = {"identity_key": obs["identity_key"], "source": obs["source"],
                                  "truth": obs["truth"], "pixel_sha256": obs["pixel_sha256"],
                                  "raw_fields": result["raw_fields"], "result": result}
            timings.append({"pipeline": kind, "block": block_index + 1, "repetition": repetition + 1,
                            "identity_key": obs["identity_key"], "source": obs["source"],
                            "inference_ms": result["inference_latency_ms"], "end_to_end_ms": result["latency_ms"]})
        fs, ds, per_field, summary = evaluate_predictions(observations, predictions)
        fs.to_csv(run / f"{kind}_field_scores.csv", index=False)
        ds.to_csv(run / f"{kind}_document_scores.csv", index=False)
        per_field.to_csv(run / f"{kind}_per_field.csv", index=False)
        summary.to_csv(run / f"{kind}_summary.csv", index=False)
        write_json(run / f"{kind}_predictions.json", predictions)
        blocks.append({"block": block_index + 1, "pipeline": kind, "repetition": repetition + 1,
                       "memory": memory(), "summary": summary.to_dict(orient="records")})
        occurrences[kind] += 1
        del pipeline, predictions, result
        gc.collect()
        torch.cuda.empty_cache()
        write_json(run / "blocks.json", blocks)
        pd.DataFrame(timings).to_csv(run / "timings.csv", index=False)
    timing = pd.DataFrame(timings)
    summary = timing.groupby(["pipeline", "source"]).agg(
        requests=("end_to_end_ms", "size"), inference_mean_ms=("inference_ms", "mean"),
        inference_median_ms=("inference_ms", "median"), end_to_end_mean_ms=("end_to_end_ms", "mean"),
        end_to_end_median_ms=("end_to_end_ms", "median"),
    ).reset_index()
    assert (summary.requests == 45).all() and len(timing) == 180
    summary.to_csv(run / "latency_summary.csv", index=False)
    ocr_fs = pd.read_csv(run / "ocr_field_scores.csv")
    donut_fs = pd.read_csv(run / "donut_field_scores.csv")
    pair = ocr_fs.merge(donut_fs, on=["identity_key", "source", "field"], suffixes=("_ocr", "_donut"), validate="one_to_one")
    pair.to_csv(run / "paired_fields.csv", index=False)
    complement = {
        "both_correct": int((pair.normalized_exact_match_ocr & pair.normalized_exact_match_donut).sum()),
        "ocr_only": int((pair.normalized_exact_match_ocr & ~pair.normalized_exact_match_donut).sum()),
        "donut_only": int((~pair.normalized_exact_match_ocr & pair.normalized_exact_match_donut).sum()),
        "both_wrong": int((~pair.normalized_exact_match_ocr & ~pair.normalized_exact_match_donut).sum()),
    }
    write_json(run / "completion.json", {
        "status": "completed", "parity_observations": len(verified),
        "matching_raw_field_predictions": 210, "matching_measured_repeats": len(timing),
        "latency_summary": summary.to_dict(orient="records"),
        "field_complementarity_development_only": complement,
        "test_evaluated": False, "policy_sha256": sha(policy_path),
    })
    print("COMPLETED", run.relative_to(ROOT), "PARITY 30/30; TIMED REPEATS 180/180", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
