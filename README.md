# SKlearnLM

An audited text snapshot of scikit-learn documentation for small transformer
training experiments.

The source is the English `1.10.dev0` documentation. It contains explanations,
API references, equations and Python examples. It is a documentation corpus,
not a conversational instruction dataset or a trained model.

## Source snapshot

[`scikit_learn_all_text.txt`](scikit_learn_all_text.txt) contains the extracted
text with page labels. The original downloaded website is excluded from Git.

Verified on 2026-10-20:

- All 1,076 document paths in the live official search index exist locally.
- 1,075 pages have readable content in the extraction. The remaining page is
  an internal contributor-link and reStructuredText substitutions file.
- All 584 generated API reference pages are present.
- All 1,601 code cells from 281 example notebooks match the extraction after
  whitespace normalization.
- The combined Python code from each of the 281 notebooks parses successfully.
  These are syntax checks; examples have not been executed.
- No UTF-8 replacement characters or null bytes were found.

The text snapshot is 7,898,411 bytes. Custom byte-level BPE measurements on this
file are 2,091,168 tokens with an 8,000-item vocabulary, 1,988,523 with 16,000,
and 1,942,178 with 32,000. These counts describe the original extraction and
depend on the tokenizer configuration.

## Intended use

This corpus supports an educational transformer training experiment. Its small
size, narrow domain and repeated API boilerplate limit generalization. It
includes development-version APIs and historical release notes.

Keep document/topic groups in one split, train the tokenizer on training data
only, and use held-out documents to assess memorization.

A structured document corpus and reproducible preparation workflow are being
added in subsequent commits.

## Attribution

Documentation and examples are from scikit-learn under BSD-3-Clause.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original preparation code
and metadata use the [BSD-3-Clause license](LICENSE).
