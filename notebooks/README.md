# Encoder–decoder training notebooks

Start with [01 — Next-word prediction](SKlearnLM_Transformer.ipynb), or
[open it in Colab](https://colab.research.google.com/github/vishal88736/SKlearnLM/blob/main/notebooks/SKlearnLM_Transformer.ipynb).

The first notebook implements the requested encoder–decoder Transformer with simple,
explicit PyTorch operations. Its 49 cells walk through BPE, teacher forcing,
sinusoidal positions, query/key/value projections, masks, encoder self-attention,
decoder self-attention, cross-attention, feed-forward layers, loss, training,
validation, checkpoints and autoregressive generation.

## Run notebook 01

Use a Python 3.10+ Jupyter kernel. The first code cell installs missing packages,
or install them first:

~~~bash
python -m pip install -r notebooks/requirements.txt
~~~

Run cells in order. Training is enabled in the configuration: running the training
cell starts the configured experiment. CUDA is selected when available; otherwise
the model uses CPU. A GPU is recommended for the default five epochs.

Defaults: 8,000 BPE vocabulary entries, width 128, four attention heads, two encoder
layers, two decoder layers, source limit 192, target limit 128 and batch size 16.
Actual vocabulary, accepted pairs, token counts and parameters are printed.

`DATA_MODE = "sentences"` prepares the same context/next-sentence pairs needed for
the next-concept comparison. It trains next-token prediction inside each target
sentence. It excludes code, headings, tables and unsuitable prose; the complete
documentation remains intact in `data/`. The tokenizer is trained only on
`train.txt`, including its code, and stays frozen.

`DATA_MODE = "all_text"` instead trains document-local full-text continuation
windows, including code. That is a distinct experiment and should not be presented
as a controlled comparison against a sentence-only concept model.

Final test evaluation is off by default. Enable it after fixing the experiment
with validation data. Normal final validation covers the full split.

## Data and artifacts

The notebook verifies the pinned dataset manifest and all file hashes. It uses
local repository files when available or downloads the matching snapshot from
commit `bb503b2b994518c4640e4d966392b03cb35b6cbe`.

Artifacts go to `runs/next_word/`. Use `SKLEARNLM_RUN_DIR` to select a fresh
experiment folder, and `SKLEARNLM_DATA_DIR` to select the existing corpus.

- `sentence_pairs.jsonl` and `pair_manifest.json`: exact accepted context/target
  pairs, split memberships and length-filter counts for the comparison.
- `tokenizer.json` and `tokenizer_metadata.json`: the training-only BPE.
- `experiment.json`: model configuration and data/tokenizer fingerprints.
- `best.pt`, `last.pt`, `history.json`, `metrics.json`: weights and actual measurements.
- `positions.png`, `causal_attention.png`, `loss_curve.png`: explanatory/measured plots.
- `next_word_inference.zip`: inference weights, tokenizer, pairs and metadata.

Keep the same pair/tokenizer configuration across initialization-seed trials.
Changing settings requires a fresh output directory. Checkpoint reload is
implemented; exact interrupted-run resumption is not.

## Checks and their limits

~~~bash
python scripts/check_training_notebook.py
python -m pip install -r notebooks/requirements-test.txt
python scripts/check_training_notebook.py --smoke
~~~

The runtime check executes all cells in a real Jupyter CPU kernel with a tiny
model and two optimizer updates on actual training pairs. It checks manual
attention against PyTorch, decoder causality, source-padding invariance,
cross-attention dependence on context, finite gradients, checkpoint reload,
generation and the export ZIP.

This is a functional smoke test, not full training, convergence evidence or GPU
validation. Published notebooks have empty outputs and contain no invented scores.

## Sources

The supplied [reference notebook](https://drive.google.com/file/d/1qyydJT5PC0eK4B0pXlbJ3N1f3gPKA6i6/view)
was inspected before writing the original training notebook. Its provenance is in
[reference_provenance.json](reference_provenance.json); it is not redistributed.

The requested architecture follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
at a small scale. The planned next-concept experiment uses the
[LCM research](https://arxiv.org/abs/2412.08821) and pretrained
[SONAR](https://github.com/facebookresearch/SONAR), as requested. It will have
outside pretrained knowledge and different losses, which the comparison must disclose.
