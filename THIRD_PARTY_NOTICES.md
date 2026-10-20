# Third-party notices

The documentation and Python examples in this repository are derived from
[scikit-learn](https://scikit-learn.org/), version `1.10.dev0`.

Upstream documentation: https://scikit-learn.org/dev/
Upstream source: https://github.com/scikit-learn/scikit-learn
Snapshot source revision linked by the API pages: `4e624cfb1`.
License source: https://github.com/scikit-learn/scikit-learn/blob/4e624cfb1/COPYING

The corpus preserves the upstream attribution in example code. The copyright
and license below apply to the scikit-learn material; the repository's root
LICENSE applies to original corpus preparation code and metadata.

This repository is an independent dataset preparation project.

```text
BSD 3-Clause License

Copyright (c) 2007-2026 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Training notebook reference

The training notebook was written after inspecting the user-supplied notebook at
https://drive.google.com/file/d/1qyydJT5PC0eK4B0pXlbJ3N1f3gPKA6i6/view.
Its conceptual organization includes tokenization, causal attention, GPT blocks,
loss, training and generation. The supplied reference links to Sebastian Raschka's
https://github.com/rasbt/LLMs-from-scratch project. Reference provenance is recorded
in notebooks/reference_provenance.json. The original reference notebook and its
book-style explanations are not redistributed; the adaptation uses newly written
code and explanations for this corpus.
