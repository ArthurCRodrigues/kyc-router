"""Measure two complete optimizer steps, retaining OOM and API failures.

Run each declared mode in a fresh process:
    .venv/bin/python -m experiments.donut_feasibility --mode full_checkpointed
    .venv/bin/python -m experiments.donut_feasibility --mode frozen_encoder
"""

import argparse
import time
import traceback

import torch
from transformers.optimization import Adafactor

from .donut_workshop import (
    ROOT, environment, load_candidate, load_observations, memory, new_run,
    read_json, setup, sha, target_ids, teacher_forcing, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full_checkpointed", "frozen_encoder", "frozen_features", "frozen_features_adafactor"], required=True)
    args = parser.parse_args()
    setup()
    run = new_run("phase4_feasibility")
    policy_path = ROOT / "configs/donut_dataset_v1.json"
    policy = read_json(policy_path)
    write_json(run / "policy.json", policy)
    if args.mode == "frozen_features":
        write_json(run / "followup_policy.json", read_json(ROOT / "configs/donut_feasibility_followup_v1.json"))
    if args.mode == "frozen_features_adafactor":
        write_json(run / "followup_policy.json", read_json(ROOT / "configs/donut_adafactor_feasibility_v1.json"))
    write_json(run / "environment.json", environment())
    result = {"mode": args.mode, "status": "started", "stage": "load",
              "policy_sha256": sha(policy_path), "completed_optimizer_steps": 0}
    print("RUN", run.relative_to(ROOT), "MODE", args.mode, flush=True)
    try:
        observations = load_observations(splits=("train",))
        processor, model = load_candidate(policy, device="cuda:0")
        obs = max(observations, key=lambda o: len(target_ids(processor, o["truth"])))
        ids = target_ids(processor, obs["truth"])
        inputs, labels = teacher_forcing(ids, "cuda:0")
        pixels = processor(images=obs["image"], return_tensors="pt").pixel_values.cuda()
        features = None
        if args.mode != "full_checkpointed":
            model.encoder.requires_grad_(False).eval()
            model.decoder.train()
            if args.mode.startswith("frozen_features"):
                with torch.no_grad():
                    features = model.encoder(pixel_values=pixels, return_dict=True).last_hidden_state.detach()
                model.encoder.to("cpu")
                del pixels
                torch.cuda.empty_cache()
        else:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        parameters = [p for p in model.parameters() if p.requires_grad]
        if args.mode == "frozen_features_adafactor":
            optimizer = Adafactor(parameters, lr=0.00005, weight_decay=0.01,
                                  scale_parameter=False, relative_step=False, beta1=None,
                                  warmup_init=False, clip_threshold=1.0)
        else:
            optimizer = torch.optim.AdamW(parameters, lr=policy["feasibility"]["learning_rate"],
                                          weight_decay=0.01, foreach=False, fused=False)
        result.update(identity_key=obs["identity_key"], split="train", source=obs["source"],
                      target_tokens=len(ids), trainable_parameters=sum(p.numel() for p in parameters),
                      gradient_checkpointing=args.mode == "full_checkpointed", steps=[])
        torch.cuda.reset_peak_memory_stats()
        for step in range(10 if args.mode == "frozen_features_adafactor" else 2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result["stage"] = f"step_{step + 1}_forward"
            if features is None:
                output = model(pixel_values=pixels, decoder_input_ids=inputs, labels=labels,
                               use_cache=False, return_dict=True)
            else:
                output = model.decoder(input_ids=inputs, encoder_hidden_states=features, labels=labels,
                                       use_cache=False, return_dict=True)
            assert torch.isfinite(output.loss)
            loss = float(output.loss.detach())
            result["stage"] = f"step_{step + 1}_backward"
            output.loss.backward()
            if args.mode == "frozen_features_adafactor":
                norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None]))
            else:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            assert torch.isfinite(norm)
            result["stage"] = f"step_{step + 1}_optimizer"
            optimizer.step()
            torch.cuda.synchronize()
            result["completed_optimizer_steps"] += 1
            row = {"step": step + 1, "loss": loss, "seconds": time.perf_counter() - start,
                   "gradient_norm_before_clip": float(norm), **memory()}
            result["steps"].append(row)
            print("STEP", row, flush=True)
            del output
        result.update(status="passed", stage="complete", optimizer_state_entries=len(optimizer.state))
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        (run / "traceback.txt").write_text(traceback.format_exc())
        print("FAILURE", result["stage"], type(error).__name__, str(error)[:450], flush=True)
    finally:
        result["memory"] = memory()
        write_json(run / "result.json", result)
        print("SAVED", run / "result.json", flush=True)


if __name__ == "__main__":
    main()
