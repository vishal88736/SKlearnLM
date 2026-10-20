#!/usr/bin/env python3
"""Validate the encoder-decoder notebook; optionally run its real Jupyter CPU or CUDA smoke path."""
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
            assert cell["execution_count"] is None and cell["outputs"] == [], "Clear published outputs"
            ast.parse("".join(cell["source"]), filename=f"{path.name}:{cell['id']}")
            count += 1
    assert {"setup", "config", "data", "tokenizer", "sentence-sections", "pairs",
            "dataset", "positions", "attention", "masks", "feedforward", "encoder",
            "decoder", "model", "model-checks", "evaluation", "training", "plots",
            "restore", "test-evaluation", "generation", "export"} <= set(ids)
    assert notebook["metadata"]["sklearnlm"]["architecture"] == "encoder_decoder"
    assert notebook["metadata"]["sklearnlm"]["objective"] == "next_token"
    return notebook, {"notebook": path.name, "cells": len(ids), "code_cells": count,
                      "static_validation": "passed"}


def execute(path, folder, device="cpu"):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    keys = ["SKLEARNLM_SMOKE", "SKLEARNLM_NO_INSTALL", "SKLEARNLM_DATA_DIR",
            "SKLEARNLM_RUN_DIR", "SKLEARNLM_SMOKE_DEVICE", "MPLBACKEND", "TOKENIZERS_PARALLELISM"]
    original = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(SKLEARNLM_SMOKE="1", SKLEARNLM_NO_INSTALL="1",
                          SKLEARNLM_DATA_DIR=str(ROOT / "data"), SKLEARNLM_RUN_DIR=str(folder), SKLEARNLM_SMOKE_DEVICE=device,
                          MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
        NotebookClient(notebook, timeout=300, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}, allow_errors=False).execute()
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return notebook


def smoke(path, folder, device="cpu"):
    import torch

    execute(path, folder, device)
    history = json.loads((folder / "history.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    checks = json.loads((folder / "model_checks.json").read_text())
    tokenizer_metadata = json.loads((folder / "tokenizer_metadata.json").read_text())
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    assert history[-1]["step"] == 2 and all(math.isfinite(row["train_loss"]) for row in history)
    assert all(checks.values())
    assert metrics["smoke_test"] is True
    assert metrics["architecture"] == "encoder_decoder"
    assert not metrics["full_training_performed"]
    for split in ("validation", "test"):
        assert math.isfinite(metrics[split]["cross_entropy"])
        assert metrics[split]["scope"] == "two_batches_only"
    assert tokenizer_metadata["trained_on"] == "train.txt"
    assert tokenizer_metadata["training_file_sha256"] == manifest["files"]["train.txt"]["sha256"]
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["experiment"]["config"]
    assert checkpoint["experiment"]["smoke_test"] and checkpoint["step"] == 2
    assert config["encoder_layers"] == config["decoder_layers"] == (2 if device == "cuda" else 1)
    assert metrics["runtime"]["device"] == device
    assert "scaler_state" in checkpoint
    if device == "cuda":
        assert torch.cuda.is_available() and metrics["runtime"]["mixed_precision"]
        assert checks["mixed_precision_forward"] is True
        assert metrics["peak_cuda_bytes"] > 0
    with zipfile.ZipFile(folder / "next_word_inference.zip") as archive:
        assert archive.testzip() is None
        assert {"inference_model.pt", "sentence_pairs.jsonl", "pair_manifest.json",
                "tokenizer.json", "experiment.json", "metrics.json"} <= set(archive.namelist())
    return {("gpu_notebook_execution" if device == "cuda" else "cpu_notebook_execution"): "passed", "optimizer_updates": 2,
            "parameters": metrics["parameters"], "decoder_causality": "passed",
            "encoder_cross_attention": "passed", "checkpoint_reload": "passed",
            "inference_bundle": "passed", "full_training_performed": False,
            "gpu_validated": device == "cuda", "runtime": metrics["runtime"],
            "peak_cuda_bytes": metrics.get("peak_cuda_bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Run two real CUDA updates using the normal model dimensions")
    parser.add_argument("--output", type=Path, help="Retain smoke artifacts for the concept notebook test")
    args = parser.parse_args()
    if args.gpu and not args.smoke:
        parser.error("--gpu requires --smoke")
    device = "cuda" if args.gpu else "cpu"
    _, report = check(args.notebook)
    if args.smoke:
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            report.update(smoke(args.notebook, args.output, device))
        else:
            with tempfile.TemporaryDirectory(prefix="sklearnlm-word-smoke-") as temporary:
                report.update(smoke(args.notebook, Path(temporary), device))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
