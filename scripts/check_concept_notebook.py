#!/usr/bin/env python3
"""Check concept notebook syntax; optionally execute a clearly labelled fixture or live SONAR smoke run."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_Next_Concept_Prediction.ipynb"


def check(path):
    notebook = json.loads(path.read_text())
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    count = 0
    for cell in notebook["cells"]:
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            ast.parse("".join(cell["source"]), filename=f"{path.name}:{cell['id']}")
            count += 1
    assert {"setup", "config", "pairs", "sonar-pipelines", "embedding-cache",
            "normalization", "dataset", "shared-attention", "shared-encoder",
            "shared-decoder", "model", "model-checks", "training", "restore",
            "test-evaluation", "word-model", "word-generation", "comparison-generation",
            "comparison-metrics", "costs", "rollout", "export"} <= set(ids)
    meta = notebook["metadata"]["sklearnlm"]
    assert meta["embedding_backend"] == "pretrained_sonar"
    assert meta["architecture"] == "encoder_decoder" and meta["requires_word_notebook"]
    return {"notebook": path.name, "cells": len(ids), "code_cells": count,
            "static_validation": "passed"}


FIXTURE = r'''
import hashlib
from types import SimpleNamespace
import torch
from torch import nn

class Tokenizer:
    def create_encoder(self, **kwargs):
        return lambda text: torch.arange(max(1, len(text.split())))

class TextToEmbeddingModelPipeline(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.tokenizer = Tokenizer()

    def predict(self, texts, **kwargs):
        # Deterministic non-semantic vectors. This does not implement SONAR.
        return torch.stack([torch.tensor(
            [byte / 127.5 - 1.0 for byte in hashlib.sha256(text.encode()).digest() * 32]
        ) for text in texts])

class EmbeddingToTextModelPipeline(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def predict(self, vectors, **kwargs):
        return ["TEST FIXTURE OUTPUT. No semantic model was run."] * len(vectors)
'''


def smoke(path, word_run, fixture, device="cpu"):
    import nbformat
    from nbclient import NotebookClient
    import torch

    keys = ["SKLEARNLM_SMOKE", "SKLEARNLM_NO_INSTALL", "SKLEARNLM_CONCEPT_FIXTURE",
            "SKLEARNLM_WORD_RUN_DIR", "SKLEARNLM_CONCEPT_RUN_DIR", "SKLEARNLM_SMOKE_DEVICE",
            "MPLBACKEND", "TOKENIZERS_PARALLELISM", "PYTHONPATH"]
    original = {key: os.environ.get(key) for key in keys}
    try:
        with tempfile.TemporaryDirectory(prefix="sklearnlm-concept-smoke-") as temporary:
            folder = Path(temporary)
            if word_run is None:
                from check_training_notebook import smoke as word_smoke, NOTEBOOK as WORD_NOTEBOOK
                word_run = folder / "next_word"
                word_run.mkdir()
                word_smoke(WORD_NOTEBOOK, word_run, device)
            concept_run = folder / "next_concept"
            os.environ.update(SKLEARNLM_SMOKE="1", SKLEARNLM_NO_INSTALL="1",
                              SKLEARNLM_CONCEPT_FIXTURE="1" if fixture else "0",
                              SKLEARNLM_WORD_RUN_DIR=str(word_run.resolve()),
                              SKLEARNLM_CONCEPT_RUN_DIR=str(concept_run), SKLEARNLM_SMOKE_DEVICE=device,
                              MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
            if fixture:
                package_root = folder / "test_fixtures"
                package = package_root / "sonar"
                (package / "inference_pipelines").mkdir(parents=True)
                (package / "__init__.py").write_text("__sklearnlm_test_fixture__ = True\n")
                (package / "inference_pipelines" / "__init__.py").write_text("")
                (package / "inference_pipelines" / "text.py").write_text(FIXTURE)
                os.environ["PYTHONPATH"] = str(package_root) + (
                    os.pathsep + original["PYTHONPATH"] if original["PYTHONPATH"] else "")
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            NotebookClient(notebook, timeout=600 if not fixture else 300, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}, allow_errors=False).execute()
            metrics = json.loads((concept_run / "metrics.json").read_text())
            comparison = json.loads((concept_run / "comparison.json").read_text())
            assert metrics["fixture_test"] is fixture and metrics["live_sonar_used"] is not fixture
            assert metrics["smoke_test"] and metrics["optimizer_updates"] == 2
            assert not metrics["full_training_performed"]
            assert metrics["runtime"]["device"] == device
            if device == "cuda":
                assert torch.cuda.is_available()
                assert metrics["peak_cuda_bytes"] > 0
            assert all(json.loads((concept_run / "model_checks.json").read_text()).values())
            for split in ("validation", "test"):
                assert math.isfinite(metrics[split]["standardized_mse"])
                assert metrics[split]["scope"] == "two_batches_only"
            assert comparison["examples"] == 2 and comparison["fixture_test"] is fixture
            assert len(set(comparison["pair_ids"])) == 2
            word_experiment = json.loads((word_run / "experiment.json").read_text())
            assert comparison["pairs_sha256"] == word_experiment["pairs_sha256"]
            for scores in comparison["scores"].values():
                assert all(math.isfinite(value) for value in scores.values())
            normalization = torch.load(concept_run / "normalization.pt", weights_only=True)
            assert normalization["info"]["fit_split"] == "train"
            model = torch.load(concept_run / "concept_inference.pt", weights_only=True)
            assert model["experiment"]["fixture_test"] is fixture
            with zipfile.ZipFile(concept_run / "next_concept_inference.zip") as archive:
                assert archive.testzip() is None and "comparison.json" in archive.namelist()
            return {("gpu_notebook_execution" if device == "cuda" else "cpu_notebook_execution"): "passed", "optimizer_updates": 2,
                    "embedding_backend": "test_fixture_not_sonar" if fixture else "pretrained_sonar",
                    "matching_pair_fingerprints": "passed", "checkpoint_reload": "passed",
                    "comparison_export": "passed", "live_sonar_weights_validated": not fixture,
                    "full_training_performed": False,
                    "gpu_concept_predictor_validated": device == "cuda",
                    "runtime": metrics["runtime"], "peak_cuda_bytes": metrics["peak_cuda_bytes"]}
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
    parser.add_argument("--gpu", action="store_true", help="Execute the concept predictor on CUDA; SONAR fixtures remain labelled")
    parser.add_argument("--live-sonar", action="store_true",
                        help="Use real SONAR assets: requires a prepared environment and about 7.6GB downloads")
    parser.add_argument("--word-run", type=Path, help="Existing two-update word smoke artifacts")
    args = parser.parse_args()
    if args.gpu and not args.smoke:
        parser.error("--gpu requires --smoke")
    device = "cuda" if args.gpu else "cpu"
    report = check(args.notebook)
    if args.smoke:
        report.update(smoke(args.notebook, args.word_run, not args.live_sonar, device))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
