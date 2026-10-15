"""Bounded Phase 4 training with real free-generation validation and evidence.

    .venv/bin/python -m experiments.train_donut

All fitting uses training identities; test loading is refused by the adapter.
Training-only features are deterministic because the encoder is frozen. The
deployed inference path never receives this cache or an identity/reference.
"""

import argparse
import gc
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers.optimization import Adafactor

from .donut_workshop import (
    ROOT, environment, evaluate_predictions, load_candidate, load_observations,
    memory, new_run, read_json, research_predict, setup, sha, target_ids,
    teacher_forcing, write_json,
)


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_features(observations, policy, run):
    processor, model = load_candidate(policy, device="cuda:0")
    encoder_hash = state_hash(model.encoder)
    features = []
    start = time.perf_counter()
    for i, obs in enumerate(observations):
        pixels = processor(images=obs["image"], return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            feature = model.encoder(pixel_values=pixels, return_dict=True).last_hidden_state.detach().cpu().clone()
        features.append(feature)
        if (i + 1) % 20 == 0:
            print("ENCODED", i + 1, "/", len(observations), flush=True)
    metadata = [{k: obs[k] for k in ("identity_key", "split", "source", "pixel_sha256")} for obs in observations]
    torch.save({"features": features, "metadata": metadata, "encoder_sha256": encoder_hash}, run / "training_features.pt")
    write_json(run / "feature_cache.json", {
        "count": len(features), "shape": list(features[0].shape), "dtype": "float32",
        "encoder_state_sha256": encoder_hash, "dataset_policy": policy,
        "seconds": time.perf_counter() - start, "metadata": metadata,
        "file_sha256": sha(run / "training_features.pt"),
    })
    del processor, model, pixels
    gc.collect()
    torch.cuda.empty_cache()
    return features, encoder_hash


def configure(policy):
    processor, model = load_candidate(policy)
    model.encoder.requires_grad_(False).eval()
    model.decoder.to("cuda:0").train()
    return processor, model


def make_optimizer(parameters, config, lr):
    tc = config["training"]
    if tc["optimizer"] == "Adafactor":
        options = dict(tc["adafactor"])
        options["eps"] = tuple(options["eps"])
        return Adafactor(parameters, lr=lr, weight_decay=tc["weight_decay"], **options)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=tc["weight_decay"],
                            betas=tuple(tc["betas"]), eps=tc["epsilon"],
                            foreach=tc["foreach"], fused=tc["fused"])


def check_gradients(parameters, config):
    if config["training"]["gradient_clip_norm"] is not None:
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip_norm"])
    else:
        norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None]))
    assert torch.isfinite(norm)
    return norm


def free_predictions(processor, model, observations, features, policy):
    model.decoder.eval()
    # Parent generation chooses its device from the first encoder parameter and
    # creates a distinct self/cross-attention cache. Temporarily co-locate the
    # frozen encoder for validation; its cached outputs still avoid recomputation.
    model.encoder.to("cuda:0").eval()
    predictions = [research_predict(processor, model, o["image"], policy, encoder_features=f)
                   for o, f in zip(observations, features)]
    model.encoder.to("cpu")
    model.decoder.train()
    torch.cuda.empty_cache()
    return predictions


def records(observations, predictions):
    return [{**{k: o[k] for k in ("identity_key", "source", "split", "pixel_sha256")},
             "truth": o["truth"], "truth_raw": o["truth_raw"], **p}
            for o, p in zip(observations, predictions)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/donut_finetuning_v2.json")
    parser.add_argument("--feature-cache", default=None)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = read_json(config_path)
    policy_path = ROOT / config["dataset_policy"]
    policy = read_json(policy_path)
    assert read_json(ROOT / config["feasibility_evidence"])["status"] == "passed"
    setup(config["seed"])
    run = new_run("phase4_finetuning")
    checkpoint = ROOT / "checkpoints/phase4_finetuning" / run.name / "best"
    checkpoint.mkdir(parents=True)
    write_json(run / "training_policy.json", config)
    write_json(run / "dataset_policy.json", policy)
    write_json(run / "environment.json", environment())
    source_hashes = {p: sha(ROOT / p) for p in [
        "experiments/train_donut.py", "experiments/donut_workshop.py", "experiments/donut_protocol.py",
        "configs/donut_dataset_v1.json", str(config_path.relative_to(ROOT)), "splits/alb_id_v1.csv"]}
    write_json(run / "source_hashes.json", source_hashes)
    for name in source_hashes:
        destination = run / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    print("RUN", run.relative_to(ROOT), flush=True)
    observations = load_observations()
    assert len(observations) == 170
    if args.feature_cache:
        cache_path = ROOT / args.feature_cache
        metadata = read_json(cache_path.parent / "feature_cache.json")
        assert sha(cache_path) == metadata["file_sha256"]
        assert metadata["dataset_policy"] == policy
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        expected = [{k: obs[k] for k in ("identity_key", "split", "source", "pixel_sha256")} for obs in observations]
        assert cached["metadata"] == expected == metadata["metadata"]
        features, encoder_hash = cached["features"], cached["encoder_sha256"]
        write_json(run / "feature_cache.json", {**metadata, "source_file": str(cache_path.relative_to(ROOT))})
    else:
        features, encoder_hash = cache_features(observations, policy, run)
    train_indices = [i for i, o in enumerate(observations) if o["split"] == "train"]
    val_indices = [i for i, o in enumerate(observations) if o["split"] == "validation"]
    assert len(train_indices) == 140 and len(val_indices) == 30
    val_obs = [observations[i] for i in val_indices]
    val_features = [features[i] for i in val_indices]

    # Separate diagnostic: initialize from the same seed/checkpoint, then discard.
    setup(config["seed"])
    processor, model = configure(policy)
    tiny_identity = min(observations[i]["identity_key"] for i in train_indices)
    tiny_indices = [i for i in train_indices if observations[i]["identity_key"] == tiny_identity]
    assert len(tiny_indices) == 2
    optimizer = make_optimizer(model.decoder.parameters(), config, config["tiny_overfit"]["learning_rate"])
    write_json(run / "status.json", {"stage": "tiny_overfit", "status": "running"})
    tiny_losses, tiny_evaluations = [], []
    for step in range(1, config["tiny_overfit"]["max_optimizer_steps"] + 1):
        i = tiny_indices[(step - 1) % len(tiny_indices)]
        inputs, labels = teacher_forcing(target_ids(processor, observations[i]["truth"]), "cuda:0")
        optimizer.zero_grad(set_to_none=True)
        output = model.decoder(input_ids=inputs, encoder_hidden_states=features[i].cuda(),
                               labels=labels, use_cache=False, return_dict=True)
        assert torch.isfinite(output.loss)
        loss = float(output.loss.detach())
        output.loss.backward()
        check_gradients(list(model.decoder.parameters()), config)
        optimizer.step()
        tiny_losses.append({"step": step, "loss": loss})
        del output
        if step % config["tiny_overfit"]["evaluate_every_steps"] == 0:
            write_json(run / "tiny_overfit.json", {"losses": tiny_losses, "evaluations": tiny_evaluations,
                                                   "stage": "before_generation", "memory": memory()})
            predictions = free_predictions(processor, model, [observations[i] for i in tiny_indices],
                                           [features[i] for i in tiny_indices], policy)
            _, _, _, summary = evaluate_predictions([observations[i] for i in tiny_indices], predictions)
            correct = int(summary.normalized_matches.sum())
            tiny_evaluations.append({"step": step, "correct_fields": correct,
                                     "predictions": records([observations[i] for i in tiny_indices], predictions)})
            print("TINY", step, "loss", round(loss, 4), "correct", correct, "/ 14", flush=True)
            write_json(run / "tiny_overfit.json", {"losses": tiny_losses, "evaluations": tiny_evaluations,
                                                   "memory": memory()})
            if correct == 14:
                break
    del processor, model, optimizer, inputs, labels
    gc.collect()
    torch.cuda.empty_cache()

    # Main run starts from the untouched parent, not the tiny diagnostic's weights.
    setup(config["seed"])
    processor, model = configure(policy)
    assert state_hash(model.encoder) == encoder_hash
    tokens = {i: target_ids(processor, o["truth"]) for i, o in enumerate(observations)}
    write_json(run / "dataset_audit.json", {
        "training_identities": 70, "training_observations": 140,
        "validation_identities": 15, "validation_observations": 30, "test_loaded": False,
        "target_lengths": {"minimum": min(map(len, tokens.values())), "maximum": max(map(len, tokens.values()))},
        "examples": [{"identity_key": observations[i]["identity_key"], "split": observations[i]["split"],
                      "source": observations[i]["source"], "target_ids": tokens[i]} for i in tokens],
    })
    tc = config["training"]
    optimizer = make_optimizer(model.decoder.parameters(), config, tc["learning_rate"])
    write_json(run / "status.json", {"stage": "main_training", "status": "running"})
    updates_per_epoch = math.ceil(len(train_indices) / tc["gradient_accumulation"])
    total_updates = updates_per_epoch * tc["max_epochs"]
    def lr_scale(step):
        if step < tc["warmup_optimizer_steps"]:
            return (step + 1) / tc["warmup_optimizer_steps"]
        return max(0.0, (total_updates - step) / (total_updates - tc["warmup_optimizer_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    rng = np.random.default_rng(config["seed"])
    history, selection = [], None
    best_key, stale = (-1.0, float("-inf")), 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, tc["max_epochs"] + 1):
        model.decoder.train()
        order = rng.permutation(train_indices).tolist()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        epoch_start = time.perf_counter()
        for position, i in enumerate(order):
            inputs, labels = teacher_forcing(tokens[i], "cuda:0")
            output = model.decoder(input_ids=inputs, encoder_hidden_states=features[i].cuda(),
                                   labels=labels, use_cache=False, return_dict=True)
            assert torch.isfinite(output.loss)
            losses.append(float(output.loss.detach()))
            (output.loss / tc["gradient_accumulation"]).backward()
            if (position + 1) % tc["gradient_accumulation"] == 0 or position + 1 == len(order):
                check_gradients(list(model.decoder.parameters()), config)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            del output
        predictions = free_predictions(processor, model, val_obs, val_features, policy)
        fs, ds, per_field, summary = evaluate_predictions(val_obs, predictions)
        em = float(summary.macro_field_em.mean())
        pooled_cer = float(fs.edit_distance.sum() / fs.reference_characters.sum())
        key = (em, -pooled_cer)
        improved = key > best_key
        row = {"epoch": epoch, "mean_training_loss": float(np.mean(losses)),
               "validation_macro_field_em": em, "pooled_cer": pooled_cer,
               "correct_fields": int(summary.normalized_matches.sum()),
               "schema_failures": int(summary.schema_failures.sum()),
               "lr_next": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - epoch_start,
               "selected_at_this_epoch": improved}
        history.append(row)
        write_json(run / f"validation_epoch_{epoch:02d}.json", records(val_obs, predictions))
        per_field.to_csv(run / f"fields_epoch_{epoch:02d}.csv", index=False)
        summary.to_csv(run / f"summary_epoch_{epoch:02d}.csv", index=False)
        if improved:
            best_key, stale = key, 0
            selection = {**row, "checkpoint": str(checkpoint.relative_to(ROOT)),
                         "encoder_state_sha256": encoder_hash, "trainable_parameters": sum(p.numel() for p in model.decoder.parameters()),
                         "training_config_sha256": sha(config_path), "dataset_policy_sha256": sha(policy_path)}
            model.save_pretrained(checkpoint, safe_serialization=True)
            processor.save_pretrained(checkpoint)
            write_json(checkpoint / "selection_metadata.json", selection)
            write_json(run / "selection.json", selection)
        else:
            stale += 1
        write_json(run / "history.json", history)
        print("EPOCH", epoch, "loss", round(row["mean_training_loss"], 4),
              "EM", f"{em:.4f}", "CER", f"{pooled_cer:.4f}", "correct", row["correct_fields"],
              "schema failures", row["schema_failures"], "BEST" if improved else "", flush=True)
        if epoch >= tc["early_stopping_min_epochs"] and stale >= tc["early_stopping_patience"]:
            break
    assert state_hash(model.encoder) == encoder_hash, "Frozen encoder changed."
    write_json(run / "completion.json", {
        "status": "completed", "epochs": len(history), "selected": selection,
        "training_seconds": time.perf_counter() - started, "memory": memory(),
        "frozen_encoder_verified": True, "tiny_overfit_correct_fields": tiny_evaluations[-1]["correct_fields"],
        "test_evaluated": False,
    })
    write_json(run / "status.json", {"stage": "complete", "status": "completed"})
    print("COMPLETE", run.relative_to(ROOT), "SELECTED", selection["epoch"], flush=True)


if __name__ == "__main__":
    main()
