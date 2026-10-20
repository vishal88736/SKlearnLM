# Training notebook

Open [SKlearnLM_Transformer.ipynb](SKlearnLM_Transformer.ipynb) in Jupyter, or
[open it in Colab](https://colab.research.google.com/github/vishal88736/SKlearnLM/blob/main/notebooks/SKlearnLM_Transformer.ipynb).

The notebook was written after inspecting the 889-cell
[user-supplied reference](https://drive.google.com/file/d/1qyydJT5PC0eK4B0pXlbJ3N1f3gPKA6i6/view).
[reference_provenance.json](reference_provenance.json) records its SHA-256 and
the core cell numbers inspected. The reference itself stays local and is not
redistributed.

## What it implements

1. Verify the existing training, validation and test text against the dataset manifest.
2. Fit byte-level BPE on training documents only, preserving code whitespace and Unicode.
3. Form non-overlapping next-token windows separately within each split.
4. Build a randomly initialized GPT with learned positional embeddings, causal
   multi-head attention, pre-normalized residual blocks, GELU and tied vocabulary weights.
5. Check shifted targets, future-token masking and finite gradients.
6. Train with AdamW, gradient accumulation, warmup/cosine scheduling, clipping,
   validation probes and early stopping.
7. Save best/latest checkpoints and restore the selected model with tokenizer/data checks.
8. Measure final validation loss and optional held-out test loss.
9. Generate documentation continuations and export weights plus the frozen tokenizer.

The default configuration targets 8,000 BPE tokens, four layers, embedding width
256, four heads and a 256-token context. The notebook prints its actual vocabulary,
token counts and model parameter count. It starts from random neural weights;
it does not load a pretrained tokenizer/model or other training text.

## Run it

Use Python 3.10+ with a Jupyter kernel. The first code cell installs missing
runtime dependencies. Alternatively:

~~~bash
python -m pip install -r notebooks/requirements.txt
~~~

Run cells in order. In the configuration cell, change `RUN_TRAINING` to `True`
when ready to start training. It is disabled by default because creating or
opening the notebook should not start a full training job. CUDA is selected
when available, otherwise CPU. Supported CUDA devices use bfloat16; other
devices use float32.

When the notebook is run alone in Colab, it downloads the text splits from the
pinned dataset commit `bb503b2b994518c4640e4d966392b03cb35b6cbe`. Local repository
data is used when present. File lengths and SHA-256 values must match the
manifest.

The test evaluation flag is separate. Enable it after validation-based model
selection. Report perplexity as next-token prediction quality under the chosen
tokenizer, not factual accuracy or instruction-following ability.

## Artifacts and repeated sessions

Artifacts go to `runs/sklearnlm/` by default, outside tracked source files.
The exact directory is printed by the notebook.

- `tokenizer.json` and `tokenizer_metadata.json` preserve the training-only BPE.
- `best.pt` and `last.pt` contain model, optimizer, configuration and fingerprints.
- `history.json` and `loss_curve.png` record measured training/validation probes.
- `metrics.json` records final evaluations that were actually run.
- `sklearnlm_inference.zip` contains weights, tokenizer, configuration and metadata.

A matching frozen tokenizer is reused. If its settings/data differ, the notebook
asks for a new run directory rather than silently invalidating a checkpoint.
To start a separate experiment, set `SKLEARNLM_RUN_DIR` before the data cell.
`SKLEARNLM_DATA_DIR` can select another explicitly prepared corpus directory.

Checkpoint loading is an inference/reload path; exact interrupted-run resumption
is not implemented. Optimizer state is retained for further development.

## Validation

Structural checks need only Python's standard library:

~~~bash
python scripts/check_training_notebook.py
~~~

The optional runtime test uses a real Jupyter kernel, a tiny CPU model and two
optimizer updates on real training windows. It checks BPE round trips, causal
masking, gradients, checkpoints, generation and the inference ZIP:

~~~bash
python -m pip install -r notebooks/requirements-test.txt
python scripts/check_training_notebook.py --smoke
~~~

For a CPU-only machine, install the matching PyTorch CPU wheel first using
[PyTorch's installation selector](https://pytorch.org/get-started/locally/), then
install the remaining test requirements.

Smoke mode uses temporary artifacts and does not modify the corpus or train a
full model. Its loss values are not convergence or model-quality results.
The notebook is committed with empty outputs; full training and GPU performance
must be measured separately.
