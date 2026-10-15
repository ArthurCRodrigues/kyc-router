"""Audit Phase 4 artifacts, executed notebooks, package and documentation.

    .venv/bin/python -m experiments.verify_phase4

Expensive GPU runs are not silently repeated. Their exact saved outputs, hashes,
counts, and source snapshots are checked; model-free unit tests run afresh.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import nbformat
import pandas as pd

from .donut_workshop import ROOT, read_json, sha, write_json


def main():
    run = ROOT / "outputs/phase4_closure" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    checks = {}
    manifest = read_json(ROOT / "configs/donut_pipeline_v1.json")
    train = ROOT / manifest["training_evidence"]
    completed = read_json(train / "completion.json")
    assert completed["status"] == "completed" and completed["frozen_encoder_verified"]
    assert completed["selected"] == manifest["selection"]
    assert not completed["test_evaluated"]
    history = pd.DataFrame(read_json(train / "history.json"))
    best = history.sort_values(["validation_macro_field_em", "pooled_cer", "epoch"], ascending=[False, True, True]).iloc[0]
    assert int(best.epoch) == manifest["selection"]["epoch"]
    assert sha(ROOT / manifest["training_configuration"]) == manifest["training_configuration_sha256"]
    assert sha(ROOT / manifest["dataset_configuration"]) == manifest["dataset_configuration_sha256"]
    assert sha(ROOT / manifest["split_manifest"]) == manifest["split_manifest_sha256"]
    for name, digest in manifest["checkpoint_files_sha256"].items():
        assert sha(ROOT / manifest["checkpoint_relative_path"] / name) == digest, name
    for name, digest in read_json(train / "source_hashes.json").items():
        assert sha(train / "sources" / name) == digest
    checks["training_and_checkpoint"] = {"epochs": len(history), "selected_epoch": int(best.epoch),
                                         "correct_fields": int(best.correct_fields), "encoder_hash_verified": True}
    comparison = ROOT / "outputs/phase4_comparison/20261015T124404754152Z"
    compared = read_json(comparison / "completion.json")
    assert compared["status"] == "completed" and not compared["test_evaluated"]
    for name, digest in read_json(comparison / "source_hashes.json").items():
        assert sha(ROOT / name) == digest, f"Changed since comparison: {name}"
        assert sha(comparison / "sources" / name) == digest
    assert compared["parity_observations"] == 30 and compared["matching_raw_field_predictions"] == 210
    assert compared["matching_measured_repeats"] == 180
    timings = pd.read_csv(comparison / "timings.csv")
    assert len(timings) == 180 and timings.groupby(["pipeline", "source"]).size().eq(45).all()
    verified = read_json(comparison / "donut_verification.json")
    retained = read_json(train / f"validation_epoch_{int(best.epoch):02d}.json")
    retained = {(r["identity_key"], r["source"]): r for r in retained}
    for record in verified:
        assert record["result"]["generated_token_ids"] == retained[record["identity_key"], record["source"]]["token_ids"]
    checks["comparison"] = compared
    notebooks = []
    execution_records = list((ROOT / "outputs/phase4_notebook_execution").glob("*/execution.json"))
    for prefix in ("05a", "05b", "05c", "05d"):
        path = next((ROOT / "notebooks").glob(f"{prefix}_*.ipynb"))
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert [c.execution_count for c in cells] == list(range(1, len(cells) + 1))
        assert not [o for c in cells for o in c.get("outputs", []) if o.output_type == "error"]
        code = "\n\n".join(c.source for c in cells)
        digest = hashlib.sha256(code.encode()).hexdigest()
        if prefix == "05a":
            record = read_json(ROOT / "outputs/phase4_foundations/20261013T130934816247Z/execution.json")
            assert record["status"] == "completed" and record["code_sha256"] == digest
        else:
            matches = [read_json(p) for p in execution_records
                       if read_json(p).get("code_sha256") == digest and read_json(p).get("status") == "completed"]
            assert matches, f"No successful execution for current notebook code: {path.name}"
        for cell in cells:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", {}):
                    assert output.metadata.get("image/png", {}).get("alt") or cell.metadata.get("alt")
        notebooks.append({"path": str(path.relative_to(ROOT)), "code_cells": len(cells), "code_sha256": digest})
    checks["notebooks"] = notebooks
    assert sum(n["code_cells"] for n in notebooks) == 47
    # Repository links include expected local-only data/evidence artifacts.
    links, broken = 0, []
    documents = [ROOT / name for name in ("README.md", "PROJECT.md", "EXPERIMENT_DESIGN.md")]
    documents += list((ROOT / "docs").rglob("*.md"))
    notebooks_all = list((ROOT / "notebooks").glob("*.ipynb"))
    for path in documents + notebooks_all:
        if path.suffix == ".ipynb":
            text = "\n".join(c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "markdown")
        else:
            text = path.read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            target = unquote(target.strip("<>").split("#")[0])
            if target:
                links += 1
                if not (path.parent / target).exists():
                    broken.append({"file": str(path.relative_to(ROOT)), "target": target})
    write_json(run / "link_check.json", {"links": links, "broken": broken})
    assert not broken, broken
    checks["documentation"] = {"local_links": links, "broken": 0}
    package = read_json(ROOT / "outputs/phase4_packaging/verification.json")
    assert package["independent_import"] and not package["gpu_allocated_on_import"] and package["threads_unchanged"]
    wheel = ROOT / "outputs/phase4_packaging/kyc_router-0.1.0-py3-none-any.whl"
    assert sha(wheel) == package["wheel_sha256"]
    for name in ("donut.py", "serialization.py"):
        assert sha(ROOT / "src/kyc_router" / name) == sha(ROOT / "outputs/phase4_packaging/extracted/kyc_router" / name)
    checks["package"] = package
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=ROOT, capture_output=True, text=True)
    (run / "unit_tests.txt").write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stderr
    match = re.search(r"Ran (\d+) tests", tests.stderr)
    assert match and int(match.group(1)) == 52
    checks["unit_tests_passed"] = int(match.group(1))
    # Compile research scripts without executing their training/measurement mains.
    for path in list((ROOT / "experiments").glob("*.py")) + list((ROOT / "src/kyc_router").glob("*.py")):
        compile(path.read_text(), str(path.relative_to(ROOT)), "exec")
    checks["python_sources_compile"] = True
    checks["git_diff_check"] = subprocess.run(["git", "diff", "--check"], cwd=ROOT).returncode == 0
    assert checks["git_diff_check"]
    files = documents + [ROOT / n["path"] for n in notebooks]
    files += list((ROOT / "configs").glob("*.json")) + list((ROOT / "src/kyc_router").glob("*.py"))
    write_json(run / "audit_manifest.json", {
        "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "test_evaluated": False,
    })
    print("PHASE 4 AUDIT PASSED", run.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
