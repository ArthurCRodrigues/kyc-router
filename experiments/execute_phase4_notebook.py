"""Execute B/C/D in a fresh .venv kernel; preserve inputs, outputs, and hashes.

    .venv/bin/python -m experiments.execute_phase4_notebook 05b_donut_dataset

Training records are replayed by C unless its explicit retraining switch is set;
the actual bounded GPU training is run by experiments.train_donut.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

from .donut_workshop import ROOT, new_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", choices=["05b_donut_dataset", "05c_donut_finetuning", "05d_donut_evaluation"])
    args = parser.parse_args()
    path = ROOT / "notebooks" / f"{args.notebook}.ipynb"
    run = new_run("phase4_notebook_execution")
    kernel = ROOT / ".cache/jupyter/kernels/kyc-router-phase4"
    kernel.mkdir(parents=True, exist_ok=True)
    write_json(kernel / "kernel.json", {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                       "display_name": "KYC Router Phase 4", "language": "python"})
    os.environ.update(JUPYTER_PATH=str(ROOT / ".cache/jupyter"), IPYTHONDIR=str(ROOT / ".cache/ipython"),
                      JUPYTER_RUNTIME_DIR=str(ROOT / ".cache/jupyter/runtime"),
                      MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PHASE4_NOTEBOOK_RUN=str(run))
    notebook = nbformat.read(path, as_version=4)
    code = "\n\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    record = {"notebook": str(path.relative_to(ROOT)), "interpreter": sys.executable,
              "code_sha256": hashlib.sha256(code.encode()).hexdigest()}
    nbformat.write(notebook, run / "input.ipynb")
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)
    print("RUN", run.relative_to(ROOT), flush=True)
    try:
        NotebookClient(notebook, timeout=1800, kernel_name="kyc-router-phase4",
                       resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress).execute()
        nbformat.validate(notebook)
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        nbformat.write(notebook, run / "executed.ipynb")
        write_json(run / "execution.json", record)
    for cell in notebook.cells:
        if cell.metadata.get("alt"):
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", {}):
                    output.metadata.setdefault("image/png", {})["alt"] = cell.metadata["alt"]
    nbformat.write(notebook, path)
    print("COMPLETED", args.notebook, flush=True)


if __name__ == "__main__":
    main()
