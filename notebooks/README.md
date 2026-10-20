# Encoder–decoder training notebooks

1. [01 — Next-word prediction](SKlearnLM_Transformer.ipynb) · [Open in Colab](https://colab.research.google.com/github/vishal88736/SKlearnLM/blob/main/notebooks/SKlearnLM_Transformer.ipynb)
2. [02 — Next-concept prediction with SONAR](02_Next_Concept_Prediction.ipynb) · [Open in Colab](https://colab.research.google.com/github/vishal88736/SKlearnLM/blob/main/notebooks/02_Next_Concept_Prediction.ipynb)

Run notebook 01 first. Notebook 02 reuses its trained model and exact sentence
pairs for the comparison. Both contain simple, explicit PyTorch attention,
encoder, decoder and training code, with explanations beside each step.


## Train on your NVIDIA GPU

Normal runs of both notebooks now require CUDA; they stop instead of silently
falling back to CPU. Use the CUDA environment below and run notebook 01 before 02.

The inspected local GPU is an RTX 4050 Laptop with 6 GB VRAM. The small default
model retains width 128, two encoder layers, two decoder layers and batch size 16.
The SONAR sentence-encoding batch is reduced to two, and its large frozen encoder
and decoder are loaded one at a time.

Create a separate Python 3.12 environment on Linux x86-64:

~~~bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r notebooks/requirements-cuda.txt
.venv/bin/python -m ipykernel install --prefix .venv --name sklearnlm-cuda --display-name "SKlearnLM CUDA"
.venv/bin/python -m jupyterlab
~~~

The CUDA requirements pin official PyTorch Linux CUDA 12.4 builds from PyPI,
with the published wheel hashes. They do not install or change the NVIDIA system driver.
Select the **SKlearnLM CUDA** kernel in Jupyter. In Colab, select a GPU runtime.

Each notebook prints the GPU name, CUDA version, VRAM and selected precision.
Training/evaluation use automatic mixed precision: bfloat16 when supported,
otherwise float16 with gradient scaling. Gradients are unscaled before clipping,
and checkpoints include the scaler state. Model weights remain float32.
Set `SKLEARNLM_PRECISION` to `fp16`, `bf16` or `fp32` before running the
configuration cells if an explicit precision is needed; the default is `auto`.

Pinned host-to-device batches support asynchronous CUDA copies. Neural training
and inference use the GPU; BPE training and text preparation remain CPU work.

Two-update CUDA checks run the normal model dimensions:

~~~bash
.venv/bin/python scripts/check_training_notebook.py --smoke --gpu --output runs/gpu-word-check
.venv/bin/python scripts/check_concept_notebook.py --smoke --gpu --word-run runs/gpu-word-check
~~~

The concept check uses explicitly labelled, non-semantic fixtures with the same
1,024-dimensional vector shape as SONAR. It validates the concept predictor on
CUDA, not the large pretrained SONAR encoder/decoder. Add `--live-sonar` only in
the prepared SONAR environment to test the downloaded pretrained assets.

The CPU smoke checks remain available for CI by omitting `--gpu`. A two-update
check is not full training or convergence evidence. Clear the smoke environment
variables before starting your normal five-epoch run.

## Run notebook 01

Use a Python 3.10+ Jupyter kernel. Its first cell installs missing runtime packages,
or install them first:

~~~bash
python -m pip install -r notebooks/requirements.txt
~~~

Run cells in order. Training is enabled: executing the training cell starts the
configured experiment. Normal training requires CUDA and uses mixed precision. Use the GPU setup above
before running the default five epochs.

Defaults: 8,000 BPE vocabulary entries, width 128, four attention heads, two encoder
layers, two decoder layers, source limit 192, target limit 128 and batch size 16.
The notebook prints actual vocabulary, accepted pairs, token counts and parameters.

The 51 cells cover teacher forcing, sinusoidal positions, explicit query/key/value
attention, masks, encoder self-attention, decoder self-attention, cross-attention,
feed-forward layers, token-weighted cross-entropy, training, validation,
checkpoints and autoregressive generation. No pretrained neural model is loaded.

`DATA_MODE = "sentences"` prepares context/next-sentence pairs for both notebooks.
It predicts each BPE token in the target sentence. Code blocks, headings, tables
and unsuitable prose are excluded from this matched experiment; the complete
documentation remains intact in `data/`. The BPE tokenizer is fit only on
`train.txt`, including its code, and stays frozen.

`DATA_MODE = "all_text"` instead trains document-local full-text continuation
windows, including code. That is a distinct experiment and should not be
presented as a controlled comparison with the sentence-only concept model.

Final test evaluation is off by default. Enable it after fixing the experiment
with validation data. Normal final validation covers the full split.

## Data and artifacts from notebook 01

The manifest and file hashes are verified. Local repository files are used when
available; standalone sessions download the matching dataset from commit
`bb503b2b994518c4640e4d966392b03cb35b6cbe`.

Artifacts go to `runs/next_word/`. Set `SKLEARNLM_RUN_DIR` to choose a fresh
experiment folder, and `SKLEARNLM_DATA_DIR` to select the existing corpus.

- `sentence_pairs.jsonl` and `pair_manifest.json`: exact accepted pairs, split
  memberships and length-filter counts.
- `tokenizer.json` and `tokenizer_metadata.json`: training-only BPE.
- `experiment.json`: model settings and data/tokenizer fingerprints.
- `best.pt`, `last.pt`, `history.json`, `metrics.json`: weights and measured results.
- `positions.png`, `causal_attention.png`, `loss_curve.png`: explanatory/measured plots.
- `next_word_inference.zip`: inference weights, tokenizer, pairs and metadata.

Keep pair/tokenizer settings fixed across initialization-seed trials. Changing
settings requires a fresh output directory. Checkpoint reload is implemented;
exact interrupted-run resumption is not.

## Run notebook 02

Finish notebook 01 in sentence mode first. In the same checkout, defaults find
`runs/next_word/`. In a separate Colab session, extract `next_word_inference.zip`
and set the path before running the second notebook:

~~~python
import os
os.environ["SKLEARNLM_WORD_RUN_DIR"] = "/content/next_word"
os.environ["SKLEARNLM_CONCEPT_RUN_DIR"] = "/content/runs/next_concept"
~~~

Use a fresh **Python 3.10–3.12 Linux x86-64** kernel for the pinned SONAR setup.
The first cell installs matching PyTorch 2.6.0, Fairseq2 0.5.2 and SONAR 0.5.0 when
needed, then asks for a kernel restart. It selects CPU or CUDA 12.4 wheels based
on GPU visibility. Override `SKLEARNLM_SONAR_VARIANT` with `cpu` or `cu124`
before that cell to choose explicitly.

Fairseq2 also requires the system library libsndfile. On Ubuntu, install
`libsndfile1` first; the notebook does this automatically in a root Colab
runtime if it is absent.

For a prepared Ubuntu CPU environment, the same setup can be performed in a terminal:

~~~bash
sudo apt-get install libsndfile1
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install fairseq2==0.5.2 fairseq2n==0.5.2+cpu --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cpu
python -m pip install -r notebooks/requirements-sonar.txt
~~~

For CUDA 12.4, replace the two CPU index suffixes with `cu124` and the binary
version with `fairseq2n==0.5.2+cu124`. Install a Jupyter kernel in this environment
if using a local notebook frontend.

The official text checkpoints total about 7.6 GB. The notebook loads the SONAR
encoder and decoder sequentially, freezes both, caches sentence embeddings, fits
median/IQR normalization on training sentences only, then trains a small
encoder–decoder to predict the next normalized sentence vector with MSE.
The frozen SONAR decoder converts the predicted vector into text.

This is an LCM-inspired **MSE baseline**, not a full paper reproduction. The
paper's Base-LCM uses a decoder-only core and studies stronger diffusion variants.
This notebook keeps the requested encoder–decoder architecture and a small corpus.
It does not implement diffusion, multilingual evaluation, instruction tuning or
learned end-of-document stopping. SONAR adds outside pretrained knowledge.

## Compare the experiments

Both trainable predictors use the same accepted pair IDs, splits, default hidden
width/depth, batch size, learning rate, seed and epoch budget. Their input/output
layers, parameter counts and effective sequence lengths still differ.

Notebook 02 writes to `runs/next_concept/`:

- `training_comparison.png`: token cross-entropy and concept MSE in separate panels.
- `comparison.json`: generated continuations on the same deterministic 32-example
  validation subset, references, oracle SONAR reconstructions, ROUGE-L F1 and
  SONAR cosine.
- `comparison.csv`: trainable/frozen parameters, optimizer updates,
  training/validation time, preprocessing measurements and shared text scores.
- `metrics.json`: native concept metrics, copy-last/training-center baselines,
  frozen SONAR parameters and model-loading costs.
- `concept_inference.pt`, `normalization.pt`, `experiment.json`,
  `sonar_provenance.json`, `next_concept_inference.zip`: reload/export artifacts.
- `sentence_embeddings.pt`: a fingerprinted local cache, excluded from the ZIP.

Token perplexity and concept MSE cannot be compared numerically. Shared text
scores are subset diagnostics, not full-split benchmarks. SONAR cosine favors the
concept model's representation. Oracle reconstruction is labelled and never
counted as a predictor.

Generation timing uses cached concept contexts and excludes model loading/source
embedding; it is not an end-to-end speed comparison. Preprocessing costs and
original cache-creation time are recorded separately.

Outside pretraining and different model capacities prevent isolating the
objective alone. Match a compute/parameter budget and repeat several seeds before
claiming an advantage. Technical correctness needs task-specific/manual checks.

SONAR's [text models have a non-commercial license](https://github.com/facebookresearch/SONAR/blob/main/NC_MODEL_LICENSE.md).
Their pretrained weights are referenced, not redistributed or committed here.

## Verification and its limits

~~~bash
python scripts/check_training_notebook.py
python scripts/check_concept_notebook.py
python -m pip install -r notebooks/requirements-test.txt
python scripts/check_training_notebook.py --smoke
python scripts/check_concept_notebook.py --smoke
~~~

The first smoke check executes all cells in a real Jupyter CPU kernel with a
tiny model and two optimizer updates on actual training pairs. It checks manual
attention against PyTorch, causality, source-padding invariance, cross-attention
dependence, gradients, checkpoint reload, generation and the ZIP.

The concept smoke check executes every cell with deterministic **non-semantic
test fixtures** replacing only SONAR. It checks two updates, train-only
normalization, matching pairs, checkpoint reload, scoring and exports. Fixture
metrics are labelled and are not SONAR/model-quality results.

A separate CI job imports the **actual pinned SONAR/Fairseq2/PyTorch packages**
and checks pipeline signatures/model cards without loading pretrained weights:

~~~bash
python scripts/check_sonar_api.py
~~~

In the prepared SONAR environment, a live two-update integration run is available:

~~~bash
python scripts/check_concept_notebook.py --smoke --live-sonar
~~~

It downloads real assets. Default fixture/API checks do not validate live
embedding/decoding, convergence or GPU performance. Published notebooks have
empty outputs, and no training/quality scores are invented.

## Sources

The supplied [reference notebook](https://drive.google.com/file/d/1qyydJT5PC0eK4B0pXlbJ3N1f3gPKA6i6/view)
informed the learning sequence. [reference_provenance.json](reference_provenance.json)
records its audit; the reference is not redistributed.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Large Concept Models](https://arxiv.org/abs/2412.08821)
- [SONAR implementation and setup](https://github.com/facebookresearch/SONAR)
