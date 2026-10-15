"""Execute 05a with this interpreter; retain successful and failed run evidence.

    .venv/bin/python experiments/run_donut_foundations.py --device cuda:0
    .venv/bin/python experiments/run_donut_foundations.py --device cpu

Prepare local weights first using prepare_donut_foundations.py. Execution is
offline, loads only one training identity, and performs no optimizer steps.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    args = parser.parse_args()
    source = ROOT / "notebooks/05a_donut_foundations.ipynb"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "outputs/phase4_foundations" / run_id
    run_dir.mkdir(parents=True)
    kernel_dir = ROOT / ".cache/jupyter/kernels/kyc-router-foundations"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "KYC Router foundations (.venv)",
        "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(ROOT / ".cache/jupyter")
    os.environ["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache/jupyter/runtime")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DONUT_DEVICE"] = args.device
    os.environ["DONUT_RUN_DIR"] = str(run_dir)
    notebook = nbformat.read(source, as_version=4)
    # Code hashes ignore output/Markdown edits; the complete input is retained too.
    code = "\n\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    provenance = {
        "run_id": run_id, "device_requested": args.device,
        "interpreter": sys.executable,
        "input_notebook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    }
    nbformat.write(notebook, run_dir / "input.ipynb")
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="kyc-router-foundations",
        resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress,
    )
    print(f"Run: {run_dir.relative_to(ROOT)}", flush=True)
    try:
        client.execute()
        nbformat.validate(notebook)
        provenance["status"] = "completed"
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        nbformat.write(notebook, run_dir / "executed.ipynb")
        (run_dir / "execution.json").write_text(json.dumps(provenance, indent=2) + "\n")
    nbformat.write(notebook, source)
    print("Completed; executed outputs saved to the notebook and run directory.")


if __name__ == "__main__":
    main()
