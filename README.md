# SKlearnLM

[![Corpus validation](https://github.com/vishal88736/SKlearnLM/actions/workflows/validate.yml/badge.svg)](https://github.com/vishal88736/SKlearnLM/actions/workflows/validate.yml)

A document-based scikit-learn documentation corpus for small transformer
training experiments. The text includes API references, explanations, Python
examples and LaTeX equations from the English `1.10.dev0` documentation.

## Training notebook

Open [the Jupyter notebook](notebooks/SKlearnLM_Transformer.ipynb) or [run it in Colab](https://colab.research.google.com/github/vishal88736/SKlearnLM/blob/main/notebooks/SKlearnLM_Transformer.ipynb).

The notebook now builds a complete **encoder–decoder Transformer** with simple, explicit PyTorch code: token embeddings, sinusoidal positions, attention masks, encoder blocks, decoder blocks and cross-attention. It includes next-token training, validation, checkpoints and generation. Training runs when its training cell is executed. The sentence-pair mode prepares a matched next-concept comparison; full-text continuation is also available. See [notebooks/README.md](notebooks/README.md) for usage and validation scope.

After completing the first notebook, open [02 — Next-concept prediction](notebooks/02_Next_Concept_Prediction.ipynb)
or [run it in Colab](https://colab.research.google.com/github/vishal88736/SKlearnLM/blob/main/notebooks/02_Next_Concept_Prediction.ipynb).
It uses frozen pretrained SONAR embeddings, predicts the next sentence vector, and
compares both models on identical held-out sentence pairs. It is an LCM-inspired
MSE experiment with outside pretrained knowledge. The guide explains setup,
artifacts and what the comparison can establish.

Both notebooks now require a **CUDA GPU** for normal training and use automatic
mixed precision. [GPU setup and Jupyter commands](notebooks/README.md#train-on-your-nvidia-gpu)
include the local RTX 4050 configuration and an isolated CUDA environment.

## Training files

| File | Purpose |
| --- | --- |
| [data/train.txt](data/train.txt) | Training text: 936 documents |
| [data/validation.txt](data/validation.txt) | Validation text: 90 documents |
| [data/test.txt](data/test.txt) | Final evaluation text: 49 documents |
| [data/corpus.txt](data/corpus.txt) | All 1,075 formatted documents; includes held-out data |
| [data/documents.jsonl](data/documents.jsonl) | One document per line with text, source, category, topic and split |
| [data/manifest.json](data/manifest.json) | Source coverage, exact file sizes, SHA-256 hashes and split details |
| [data/audit_report.json](data/audit_report.json) | Local checks, including original notebook preservation |
| [scikit_learn_all_text.txt](scikit_learn_all_text.txt) | Original audited extraction, retained for comparison |

Use `data/train.txt` to train a tokenizer and model. The complete corpus and
original extraction contain validation/test documents and must not be used as
training inputs when reporting results on these splits.

The split is approximately **90% / 5% / 5% by text characters**. Document counts
differ because whole topic families stay together. Related API references,
guides and examples use a shared topic group when the filename mapping
identifies their relationship. Split membership is reproducible with seed 42.

## Text format

Documents retain Markdown headings, numbered/bulleted lists, typed parameter
descriptions, table columns, inline identifiers, fenced code blocks and
original LaTeX equations. Figure alt text is included when available. Images
and executable website widgets are excluded; links retain their visible text
and each page's source URL is stored in JSONL metadata.

Each document ends with the boundary token `<|endoftext|>`:

~~~text
# LinearRegression

Parameters:

**fit_intercept** : bool, default=True

Whether to calculate the intercept for this model.

<|endoftext|>
~~~

This is an abbreviated format example. Actual records retain the complete page.
Code blocks are labelled `python`, `pycon`, `text` or their source language.
Python indentation, code-internal whitespace and equation notation are retained.

Register `<|endoftext|>` as a special token before training your tokenizer.
Chunk or pack documents only after splitting. Do not lowercase, collapse code
indentation, strip Markdown fences or fit BPE on validation/test data.

## Reproduce and check

Python 3.10 or newer is sufficient; no third-party packages are required.

Validate the published files:

~~~bash
python -m unittest discover -s tests -v
python scripts/audit_corpus.py
~~~

Rebuild from the published JSONL without downloading the website:

~~~bash
python scripts/prepare_corpus.py --from-jsonl data/documents.jsonl --output rebuilt
python scripts/audit_corpus.py --data rebuilt
~~~

Re-extract from the original local HTML snapshot when it is available:

~~~bash
python scripts/prepare_corpus.py --html-root scikit-learn-docs --output data
python scripts/audit_corpus.py --html-root scikit-learn-docs --report data/audit_report.json
~~~

The large downloaded website is ignored by Git. Re-extraction requires that
snapshot, including its `searchindex.js`. The JSONL rebuild route works from a
normal Git clone.

## Verified coverage

The local download was compared with the official development documentation
search index on 2026-10-20: all **1,076 indexed page paths** matched. The text
corpus contains **1,075 readable pages**; the omitted page consists of internal
contributor-link definitions and reStructuredText substitutions.

Local audits verified:

- All **584 API reference pages** and **3,254 indexed Python API entries** have
  their target pages and names in the formatted corpus.
- All **1,601 code cells from 281 example notebooks** occur in the formatted
  text after whitespace normalization.
- All 281 notebooks' combined Python code passes a syntax check. The examples
  have not been executed.
- Text/metadata checksums, document boundaries, split membership, topic-group
  separation and byte-for-byte offline regeneration pass.
- No UTF-8 replacement characters or null bytes occur in document text.

CI runs the formatter tests and offline corpus audit. Original-HTML and
notebook comparisons are local checks recorded in the audit report.

## Training limits

The corpus contains roughly two million BPE tokens, depending on the final
tokenizer and vocabulary. Previously measured counts in the original extraction
do not describe the formatted files or a tokenizer trained only on `train.txt`.
No trained tokenizer, neural model, loss curve or model-quality result is included.

This is a narrow technical corpus for educational pretraining. It includes
development-version APIs, historical release notes, developer and project pages.
It is not an instruction/chat dataset. Repeated API descriptions remain in
context rather than being removed from individual reference pages.

Topic grouping is a filename heuristic. The audit detects **28 long text
blocks shared across splits**, mostly generic API prose. Disjoint group IDs
do not prove that every sentence or code pattern is absent from the other
splits. A topic-held-out evaluation also measures transfer to unseen families
and should not be described as an ordinary random-sample benchmark.

See [docs/FORMAT.md](docs/FORMAT.md) for the record schema and split details.

## Attribution

Documentation and examples are derived from scikit-learn under BSD-3-Clause.
The snapshot's generated API pages predominantly link source revision
`4e624cfb1`. This project is independent of scikit-learn.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original preparation code
and metadata use the [BSD-3-Clause license](LICENSE).
