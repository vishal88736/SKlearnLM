# Corpus format

## JSONL records

`data/documents.jsonl` is the canonical structured dataset. Each UTF-8 line is a
complete JSON object. The `text` field contains the formatted page, without an
end-of-document token; plain-text exports append one boundary token per page.

| Field | Meaning |
| --- | --- |
| `id` | First 20 hexadecimal characters of SHA-256(source_path) |
| `title` | Main visible page heading |
| `source_path` | HTML path within the saved documentation |
| `source_url` | Corresponding official development-documentation URL |
| `version` | Snapshot version, `1.10.dev0` |
| `category` | API reference/index, guide, example, release notes, developer docs or project |
| `topic` | Topic inferred from the path and a small alias map |
| `group_id` | Topic family used to keep related pages in one split |
| `split` | `train`, `validation` or `test` |
| `text` | Full page as Markdown with fenced code and LaTeX equations |
| `text_sha256` | SHA-256 of the UTF-8 `text` value |
| `code_block_count` | Number of retained source HTML `pre` blocks |
| `license` | `BSD-3-Clause` |

Reading documents without mixing splits:

~~~python
import json
from pathlib import Path

with Path("data/documents.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        document = json.loads(line)
        if document["split"] == "train":
            text = document["text"]
            # Feed this document to the training tokenizer/model pipeline.
~~~

## Plain text and boundaries

`corpus.txt` contains every formatted page in source-path order. The three split
files contain the corresponding documents in a seeded shuffled order.

Each page is followed by a standalone `<|endoftext|>` marker. Register that marker
as a single special token; it is metadata for the model's document boundary.
It is never present inside a document's original text.

The corpus does not add invented user/assistant conversations, summaries,
answers or task labels. Descriptive text, code and displayed output remain in
their original page context.

## Split policy

Core topic families are candidates for validation and test holdouts. Each
holdout aims for 5% of the total corpus's text characters and includes at least
three topic groups. The remaining families, project/developer pages and
historical notes are assigned to training. This targets a 90/5/5 size split,
not equal percentages of document counts.

The default validation topics are datasets, Gaussian processes and multioutput.
The default test topics are feature extraction, multiclass and SVM. Topic aliases
connect obvious related folders, such as text/feature_extraction and
sgd/linear_model. This is a reproducible heuristic, not a semantic deduplication
algorithm.

Before tokenization, keep these files separate. Train BPE on `train.txt` only,
encode held-out files with that frozen tokenizer, then form sequences within
each split. Chunking the complete corpus before a random split can leak nearby
or repeated page content into evaluation.

## Preservation and exclusions

- Source headings, signatures, parameters and tables receive structural markers.
- Python indentation and code-internal spacing are preserved.
- Equations retain the source's LaTeX delimiters.
- Figure descriptions come only from existing alt text, without image inference.
- Navigation, download buttons, SVG/HTML estimator widgets and empty/redirect
  pages are omitted.
- External papers, datasets, source implementations and other scikit-learn
  versions are not downloaded by the preparation script.

Generic API boilerplate remains; the audit reports long-block overlap across
splits. See `data/audit_report.json` and `data/manifest.json` for the exact checks,
counts, limitations and SHA-256 values.
