#!/usr/bin/env python3
"""Check the notebook structure; optionally execute its two-update CPU smoke path."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "SKlearnLM_Transformer.ipynb"


def check(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)), "Duplicate cell IDs"
    count = 0
    for cell in notebook["cells"]:
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == [], "Published outputs must be cleared"
            ast.parse("".join(cell["source"]), filename=f"{path.name}:{cell['id']}")
            count += 1
    assert {"setup", "config", "data", "tokenizer", "dataset", "model",
            "model-checks", "training-helpers", "train-function", "plots",
            "restore", "test-evaluation", "generation", "export"} <= set(ids)
    assert notebook["metadata"]["sklearnlm"]["reference_cells"] == 889
    assert not notebook["metadata"]["sklearnlm"]["training_enabled_by_default"]
    return notebook, {"notebook": path.name, "cells": len(ids), "code_cells": count,
                      "static_validation": "passed"}


def smoke(path):
    import nbformat
    from nbclient import NotebookClient
    import torch

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    keys = ["SKLEARNLM_SMOKE", "SKLEARNLM_NO_INSTALL", "SKLEARNLM_DATA_DIR",
            "SKLEARNLM_RUN_DIR", "MPLBACKEND", "TOKENIZERS_PARALLELISM"]
    original = {key: os.environ.get(key) for key in keys}
    try:
        with tempfile.TemporaryDirectory(prefix="sklearnlm-smoke-") as temporary:
            os.environ.update(
                SKLEARNLM_SMOKE="1", SKLEARNLM_NO_INSTALL="1",
                SKLEARNLM_DATA_DIR=str(ROOT / "data"), SKLEARNLM_RUN_DIR=temporary,
                MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false",
            )
            NotebookClient(
                notebook, timeout=240, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
            ).execute()
            folder = Path(temporary)
            history = json.loads((folder / "history.json").read_text())
            metrics = json.loads((folder / "metrics.json").read_text())
            tokenizer_metadata = json.loads((folder / "tokenizer_metadata.json").read_text())
            manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
            assert len(history) == 2 and history[-1]["step"] == 2
            assert all(math.isfinite(row["train_loss"]) for row in history)
            assert metrics["smoke_test"] is True
            assert math.isfinite(metrics["validation"]["cross_entropy"])
            assert math.isfinite(metrics["test"]["cross_entropy"])
            assert metrics["validation"]["scope"] == "two_batches_only"
            assert metrics["test"]["scope"] == "two_batches_only"
            assert tokenizer_metadata["trained_on"] == "train.txt"
            assert tokenizer_metadata["training_file_sha256"] == manifest["files"]["train.txt"]["sha256"]
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            assert checkpoint["smoke_test"] is True and checkpoint["step"] > 0
            assert checkpoint["config"]["n_layers"] == 2
            assert checkpoint["config"]["vocab_size"] == 512
            with zipfile.ZipFile(folder / "sklearnlm_inference.zip") as archive:
                assert archive.testzip() is None
                assert set(archive.namelist()) == {
                    "inference_model.pt", "tokenizer.json", "tokenizer_metadata.json",
                    "config.json", "data_manifest.json", "metrics.json",
                }
            return {
                "cpu_notebook_execution": "passed", "optimizer_updates": 2,
                "parameters": metrics["parameters"],
                "checkpoint_reload": "passed", "inference_bundle": "passed",
                "full_training_performed": False,
                "gpu_validated": False,
            }
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    _, report = check(args.notebook)
    if args.smoke:
        report.update(smoke(args.notebook))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
