<p align="center">
  <img width="480" src="assets/banner.png" alt="NanoVDR"/>
</p>

<h3 align="center">Small retrievers for visual documents,<br>trained by aligning directly to a frozen VLM's embedding space</h3>

<p align="center">
  <a href="https://huggingface.co/nanovdr">Models</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/nanovdr/NanoVDR-Demo">Demo</a> &nbsp;|&nbsp;
  <a href="#write-ups">Write-ups</a> &nbsp;|&nbsp;
  <a href="#papers">Papers</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/nanovdr/NanoVDR-Train">Data</a> &nbsp;|&nbsp;
  <a href="#evaluation-harness">Eval harness</a>
</p>

---

## One method, two towers

Visual document retrieval pairs a **text query** with a **document image**. Every
model in this repository is trained the same way: a frozen vision-language
teacher encodes the input, a small student learns to reproduce that embedding,
and the loss is a direct discrepancy between the two representations. No
relevance labels, no negative mining, no contrastive sampling at student
training time.

Because the teacher targets are cached once, the two towers are independent.
You can train either alone and pair it with the teacher for the other side, or
train both and drop the teacher entirely.

| | encodes | output geometry | replaces the teacher at |
|---|---|---|---|
| **Query tower** | text | single vector **or** multi vector | query time |
| **Doc tower** | page image | single vector | indexing time |

Multi-vector output is supported on the query side only. A single-vector
objective is the one-atom special case of the multi-vector one, so both are
entries in the same registry (`nanovdr/align/`) rather than separate codepaths.

> 🚧 **Multi-vector query towers are under construction.** The encoder and its
> transport objective are in the code already; checkpoints arrive with
> NanoVDR-v2.

```
                       frozen teacher (Qwen3-VL-Embedding)
                        /                            \
              target vector                      target vector
                    |                                  |
              [ doc tower ]                      [ query tower ]
              page image                         text query
                    \__________ dot product / MaxSim _______/
                          (teacher discarded at deployment)
```

## Any half of the teacher, swapped

Both towers are trained into the same frozen teacher's embedding space, so
either one is a drop-in replacement for the corresponding half of that teacher.
All four combinations retrieve; they differ in what you no longer have to run.

|  | teacher documents | **student documents** |
|---|---|---|
| **teacher queries** | 71.05 — the ceiling, 8B on both sides | 65.02 — indexing 7x cheaper, teacher still runs per query |
| **student queries** | 66.36 — queries encode on one CPU thread, index built once by the teacher | **61.74 — no teacher anywhere** |

<sub>Average NDCG@5 over ViDoRe v1+v2+v3, against Qwen3-VL-Embedding-8B.</sub>

This is the point of aligning to a frozen space rather than training a new one:
a student is interchangeable with the half of the teacher it replaces, so the
two towers can be adopted separately and in either order.

## Naming

```
NanoVDR-<Q|D>-<variant>-<teacher>-<width>[-ML]
```

**A pair is valid when the teacher and the width both match.** The teacher
fixes the embedding space; the width fixes which part of it is targeted, since
a Matryoshka teacher can be aligned to at more than one width. The variant and
`-ML` describe how a tower was built rather than where it lands, so they never
affect pairing: `-ML` marks the multilingual training mixture, added because
the older `-Multi` read as multi-vector.

## Models

Retention is measured against **that model's own teacher**, so the column is
not comparable across teacher rows.

### Query towers

| Model | Backbone | Params | Teacher | Width | v1 | v2 | v3 | CPU latency |
|---|---|---|---|---|---|---|---|---|
| [NanoVDR-Q-DistilBERT-Qwen3VL8B-4096-ML](https://huggingface.co/nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL8B-4096-ML) ⭐ | DistilBERT | 70M | 8B | 4096 | 84.68 | 64.30 | 50.09 | 51 ms |
| [NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML](https://huggingface.co/nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML) | DistilBERT | 69M | 2B | 2048 | 82.2 | 61.9 | 46.5 | 51 ms |
| [NanoVDR-Q-BERT-Qwen3VL2B-2048-ML](https://huggingface.co/nanovdr/NanoVDR-Q-BERT-Qwen3VL2B-2048-ML) | BERT-base | 112M | 2B | 2048 | 82.5 | 62.8 | 47.5 | 101 ms |
| [NanoVDR-Q-ModernBERT-Qwen3VL2B-2048-ML](https://huggingface.co/nanovdr/NanoVDR-Q-ModernBERT-Qwen3VL2B-2048-ML) | ModernBERT | 151M | 2B | 2048 | 82.2 | 63.1 | 47.1 | 109 ms |
| [NanoVDR-Q-DistilBERT-Qwen3VL2B-2048](https://huggingface.co/nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL2B-2048) | DistilBERT | 69M | 2B | 2048 | 82.2 | 60.5 | 43.5 | 51 ms |
| [NanoVDR-Q-BERT-Qwen3VL2B-2048](https://huggingface.co/nanovdr/NanoVDR-Q-BERT-Qwen3VL2B-2048) | BERT-base | 112M | 2B | 2048 | 82.1 | 62.2 | 44.7 | 101 ms |
| [NanoVDR-Q-ModernBERT-Qwen3VL2B-2048](https://huggingface.co/nanovdr/NanoVDR-Q-ModernBERT-Qwen3VL2B-2048) | ModernBERT | 151M | 2B | 2048 | 82.4 | 61.5 | 44.2 | 109 ms |

The 8B row is scored under query-side isolation; the 2B rows are as originally
published, and the two are not comparable across teachers. Latency is a single
query on one CPU thread including tokenisation, and tracks the backbone only:
the width of the output head is marginal beside it.

### Document towers

| Model | Visual + text backbone | Params | Teacher | Width | Tiles | v1 | v2 | v3 | Avg |
|---|---|---|---|---|---|---|---|---|---|
| [NanoVDR-D-HiRes-Qwen3VL8B-4096](https://huggingface.co/nanovdr/NanoVDR-D-HiRes-Qwen3VL8B-4096) ⭐ | InternViT-300M + ModernBERT-base | 457M | 8B | 4096 | 6 | 82.81 | 55.34 | 47.07 | **61.74** |
| [NanoVDR-D-Fast-Qwen3VL8B-4096](https://huggingface.co/nanovdr/NanoVDR-D-Fast-Qwen3VL8B-4096) | InternViT-300M + ModernBERT-base | 457M | 8B | 4096 | 2 | 81.34 | 54.95 | 43.66 | 59.98 |

Every model here was renamed on 2026-08-11 to make the pairing rule readable
from the name; the old links redirect and each card records its former name.

---

## Install

```bash
pip install -e .          # add [train] or [eval] for the optional extras
```

## Quick start

Every released checkpoint also runs straight from the Hub, with no dependency
on this repository. Both towers are one call, and it is the same call.

```python
from sentence_transformers import SentenceTransformer      # >= 5.4 for the doc tower

doc = SentenceTransformer("nanovdr/NanoVDR-D-HiRes-Qwen3VL8B-4096", trust_remote_code=True)
query = SentenceTransformer("nanovdr/NanoVDR-Q-DistilBERT-Qwen3VL8B-4096-ML")

doc_emb = doc.encode(pages)                                # PIL pages -> (N, 4096)
q_emb = query.encode(["What was the revenue growth in Q3 2024?"])

scores = q_emb @ doc_emb.T
```

Through `transformers` instead, which has no version floor and returns the same
vectors:

```python
from transformers import AutoModel, AutoImageProcessor

doc = AutoModel.from_pretrained("nanovdr/NanoVDR-D-HiRes-Qwen3VL8B-4096", trust_remote_code=True).eval()
proc = AutoImageProcessor.from_pretrained("nanovdr/NanoVDR-D-HiRes-Qwen3VL8B-4096", trust_remote_code=True)
doc_emb = doc.encode(pages, proc, batch_size=4)
```

Both sides carry the same teacher and width, `Qwen3VL8B-4096`, so they pair.
`-ML` describes the query tower's training mixture, not the space it targets,
so it plays no part in the pairing rule. Swap either side for the teacher and
retrieval still works; see the matrix above.

---

## Training

All three training entry points read the same config schema and differ only in
which tower they instantiate.

```bash
# query tower, single vector
python -m nanovdr.train.query --config configs/query/distilbert_8b.yaml

# query tower, multi vector (late interaction)
python -m nanovdr.train.query --config configs/query/distilbert_8b_ot.yaml

# document tower
torchrun --nproc_per_node=2 -m nanovdr.train.doc --config configs/doc/hires_8b.yaml
```

Teacher targets are cached once beforehand and reused by every run, which is
what lets the two towers train in parallel:

```bash
nanovdr-cache --data $DATA_ROOT/base_711k --out $CACHE_ROOT/teacher_8b/base_711k.h5 --kind image
nanovdr-cache --data $DATA_ROOT/queries   --out $CACHE_ROOT/teacher_8b/queries.h5   --kind query
```

Caching the 1.20M-image and 1.49M-query mixture against the 8B teacher took
99.5 H200-GPU-hours in total. It is a one-time cost: every checkpoint and
every ablation in the paper reuses the same cache, and the teacher is never
loaded during student training.

### Alignment objectives

The objective is the only place the method varies. `nanovdr/align/` holds one
registry for both geometries:

| Key | Geometry | Notes |
|---|---|---|
| `cosine` | single | `1 - <s, t>`; the default everywhere on the single-vector side |
| `ot` | multi | entropic Sinkhorn transport between the two token measures |

Every objective consumes only the student representation and its cached
teacher target. None of them needs documents, negatives, or relevance labels
at training time, which is what lets the two towers train independently.

```python
from nanovdr.align import get_align_loss, available_align_losses

loss = get_align_loss("ot", geometry="multi", eps=0.05, n_iter=50)
available_align_losses(geometry="single")   # ['cosine']
```

Registering a new objective is one decorator; nothing else changes.

---

## Evaluation harness

`eval/` reproduces **twelve publicly released retrievers plus the teacher**
under one protocol: same evaluation driver, same input formatting, same
deduplication, same metric code. It reports retrieval quality *and* deployment
cost side by side.

```bash
nanovdr-eval --doc-model nanovdr/NanoVDR-D-HiRes-Qwen3VL8B-4096 \
             --teacher-cache $CACHE_ROOT/teacher_8b/eval \
             --benchmarks v1 v2 v3 --out results.json
```

```python
from nanovdr.evaluation import profile_model
profile_model(doc_tower, pages, batch_size=8)
# {'pages_per_second': 36.82, 'peak_vram_gb': 3.07, 'index_gb_per_million': 16.4, ...}
```

Measured per model: NDCG@5 on ViDoRe v1/v2/v3, query latency, document
throughput, peak VRAM, index size per million documents, and CPU scoring
latency at 10K candidates.

The harness carries the per-baseline handling that these models need in order
to reproduce their published behaviour — float32 for BiModernVBERT, the
released chat-template path with `use_cache=False` for DSE-Qwen2, the native
flash-attention path for InternViT, and so on. If you only take one thing from
this repository, take this.

`nanovdr-eval` writes per-dataset JSON, so a reported number can be traced to
the dataset it came from without rerunning the sweep.

---

## Repository layout

```
nanovdr/
├── align/            the objective registry; the only place the method varies
│   ├── registry.py     Repr, AlignLoss, get_align_loss
│   ├── single.py       cosine
│   └── multi.py        ot (Sinkhorn), chamfer, coverage
├── towers/
│   ├── query.py        text -> single vector or token set
│   ├── doc.py          page image -> single vector
│   ├── modeling_doc.py   standalone definition, also shipped inside every
│   │                     Hub checkpoint and loaded by trust_remote_code
│   └── processing_doc.py standalone image processor, likewise shipped; the
│                         one home for page tiling
├── heads.py          SingleVectorHead / MultiVectorHead
├── teacher.py        one-off target precomputation  (nanovdr-cache)
├── data.py           mixtures of datasets paired with cached targets
├── tiling.py         re-export of the tiling rule from processing_doc
├── scoring.py        dot product | MaxSim, dispatched on geometry
├── evaluation.py     ViDoRe scoring and deployment profiling  (nanovdr-eval)
├── config.py         YAML loading with ${VAR} expansion
└── train/
    ├── engine.py       the loop, shared by both towers
    ├── doc.py          entry point
    └── query.py        entry point
configs/              the four released recipes
packaging/            training checkpoint -> Hub package, and its verifier
tests/                objective registry and preprocessing tests, no GPU needed
```

A page does not reach the model as one 448px view. It is cut into
aspect-ratio-matched tiles plus a thumbnail, zero-padded to a fixed budget and
paired with a mask, and that rule lives in `processing_doc.py` alone so it
cannot drift between training, evaluation and release. `tests/test_processing.py`
asserts the processor is bit-identical to the preprocessing the released weights
were trained on, across five page aspect ratios.

`packaging/verify_doc_package.py` is how we established that the released
weights reproduce our internal evaluation:

```
500 pages of ViDoRe arxivqa, NanoVDR-D-HiRes-Qwen3VL8B-4096
cosine(student page, teacher page)      : 0.7959 mean
NDCG@5  teacher queries x teacher pages : 86.91
NDCG@5  teacher queries x student pages : 83.07   (95.6% retention)
```

## Write-ups

- [**Distilling the Document Tower**](https://huggingface.co/spaces/nanovdr/distilling-the-document-tower). How the 527M document tower was
  distilled from an 8B teacher: the teacher and data choices, the architecture,
  fourteen ablations, and the five things that did not work.

## Papers

- **NanoVDR**, single-vector query tower distilled from a 2B teacher.
  [arXiv:2603.12824](https://arxiv.org/abs/2603.12824)
- **DistilVDR**, both towers distilled from an 8B teacher, end-to-end and
  teacher-free at deployment. Preprint on arXiv; identifier being added here as
  soon as it is announced.
- **NanoVDR-v2**, multi-vector query tower. 🚧 Under construction.

## Citation

```bibtex
@article{nanovdr2026,
  title   = {NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M
             Text-Only Encoder for Visual Document Retrieval},
  author  = {Liu, Zhuchenyang and Zhang, Yao and Xiao, Yu},
  journal = {arXiv preprint arXiv:2603.12824},
  year    = {2026}
}

@article{distilvdr2026,
  title   = {DistilVDR: A Compact End-to-End Visual Document Retriever
             via Dual-Student Distillation},
  author  = {Liu, Zhuchenyang and Wang, Ziyi and Zhang, Yao and Xiao, Yu},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT for the code. Model weights and training data follow the licences of their
respective upstream sources; see `docs/DATA.md`.
