#!/usr/bin/env python3
"""Import the actual pinned SONAR runtime and check the API used by the notebook, without weights."""
import hashlib
import importlib.metadata
from importlib.resources import files
import inspect
import json

import torch
import sonar
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)

assert not getattr(sonar, "__sklearnlm_test_fixture__", False), "A real SONAR installation is required"
expected = {"torch": "2.6.0", "torchaudio": "2.6.0", "fairseq2": "0.5.2",
            "fairseq2n": "0.5.2", "sonar-space": "0.5.0"}
versions = {name: importlib.metadata.version(name) for name in expected}
assert all(versions[name].split("+")[0] == version for name, version in expected.items())
for cls, constructor_names, predict_names in [
    (TextToEmbeddingModelPipeline, {"encoder", "tokenizer", "device", "dtype"},
     {"input", "source_lang", "batch_size", "max_seq_len"}),
    (EmbeddingToTextModelPipeline, {"decoder", "tokenizer", "device", "dtype"},
     {"inputs", "target_lang", "batch_size"}),
]:
    assert constructor_names <= set(inspect.signature(cls.__init__).parameters)
    assert predict_names <= set(inspect.signature(cls.predict).parameters)
    assert issubclass(cls, torch.nn.Module)
card_hashes = {}
for card in ("text_sonar_basic_encoder", "text_sonar_basic_decoder"):
    payload = files("sonar.cards").joinpath(card + ".yaml").read_bytes()
    assert b"https://dl.fbaipublicfiles.com/SONAR/" in payload
    card_hashes[card] = hashlib.sha256(payload).hexdigest()
print(json.dumps({"real_sonar_import": "passed", "pipeline_api": "passed",
                  "versions": versions, "model_card_sha256": card_hashes,
                  "pretrained_weights_loaded": False,
                  "live_embedding_or_decoding_validated": False}, indent=2))
