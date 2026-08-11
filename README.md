<p align="center">
  <img width="480" src="https://huggingface.co/nanovdr/NanoVDR-S-Multi/resolve/main/banner.png" alt="NanoVDR"/>
</p>

<h3 align="center">Small retrievers for visual documents,<br>trained by aligning directly to a frozen VLM's embedding space</h3>

<p align="center">
  <a href="https://huggingface.co/nanovdr">Models</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/nanovdr/NanoVDR-Demo">Demo</a> &nbsp;|&nbsp;
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

<!-- TODO: architecture figure — reuse figs/arch.tex from the DistilVDR paper,
     redrawn to show the two towers as swappable modules rather than one system. -->

### Configuration axes

Everything released here is a point in the same space:

```yaml
tower:      query | doc
geometry:   single | multi          # doc tower: single only
teacher:    qwen3-vl-embedding-2b   # 2048-d
            qwen3-vl-embedding-8b   # 4096-d
backbone:   distilbert | bert-base | modernbert          # query
            internvit-300m + modernbert-base             # doc
align:      cosine | ot | chamfer | coverage | ...       # see registry
```

---

## Models

Retention is measured against **that model's own teacher**, so the column is
not comparable across teacher rows.

### Query tower, single vector

| Model | Backbone | Params | Teacher | ViDoRe v1 | v2 | v3 | Retention | CPU latency |
|---|---|---|---|---|---|---|---|---|
| [NanoVDR-S](https://huggingface.co/nanovdr/NanoVDR-S) | DistilBERT | 69M | 2B | 82.2 | 60.5 | 43.5 | 92.4% | 51 ms |
| [NanoVDR-M](https://huggingface.co/nanovdr/NanoVDR-M) | BERT-base | 112M | 2B | 82.1 | 62.2 | 44.7 | 94.0% | 101 ms |
| [NanoVDR-L](https://huggingface.co/nanovdr/NanoVDR-L) | ModernBERT | 151M | 2B | 82.4 | 61.5 | 44.2 | 93.4% | 109 ms |
| [NanoVDR-S-Multi](https://huggingface.co/nanovdr/NanoVDR-S-Multi) ⭐ | DistilBERT | 69M | 2B | 82.2 | 61.9 | 46.5 | 95.1% | 51 ms |
| [NanoVDR-M-Multi](https://huggingface.co/nanovdr/NanoVDR-M-Multi) ⭐ | BERT-base | 112M | 2B | 82.5 | 62.8 | 47.5 | 96.4% | 101 ms |
| [NanoVDR-L-Multi](https://huggingface.co/nanovdr/NanoVDR-L-Multi) ⭐ | ModernBERT | 151M | 2B | 82.2 | 63.1 | 47.1 | 96.0% | 109 ms |

`-Multi` denotes the **multilingual** training mixture, not multi-vector output.

### Query tower, multi vector (late interaction)

NanoVDR-v2, the multi-vector query tower, is **coming soon** — the encoder and
its alignment objective are in `nanovdr/` already; checkpoints and numbers land
with the paper.

### Doc tower, single vector

| Model | Visual + text backbone | Params | Teacher | Tiles | ViDoRe v1 | v2 | v3 | Avg |
|---|---|---|---|---|---|---|---|---|
| [NanoVDR-D-HiRes](https://huggingface.co/nanovdr/NanoVDR-D-HiRes) ⭐ | InternViT-300M + ModernBERT-base | 457M | 8B | 6 | 82.81 | 55.34 | 47.07 | **61.74** |
| [NanoVDR-D-Fast](https://huggingface.co/nanovdr/NanoVDR-D-Fast) | InternViT-300M + ModernBERT-base | 457M | 8B | 2 | 81.34 | 54.95 | 43.66 | 59.98 |

Paired with the 70M single-vector query tower distilled from the same 8B
teacher, these give an end-to-end system that never runs the teacher at
deployment: **61.74 average NDCG@5, 86.9% of the 8B teacher, one 4096-d vector
per page.**

---

## Quick start

Every released checkpoint runs from the Hub today, with no dependency on this
repository.

```python
from transformers import AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer

# document tower: page image -> one 4096-d vector
doc = AutoModel.from_pretrained("nanovdr/NanoVDR-D-HiRes", trust_remote_code=True).eval()
proc = AutoImageProcessor.from_pretrained("nanovdr/NanoVDR-D-HiRes", trust_remote_code=True)
doc_emb = doc.encode(pages, proc, batch_size=4)

# query tower: text -> one vector in the same space
query = SentenceTransformer("nanovdr/NanoVDR-S-Multi")
q_emb = query.encode(["What was the revenue growth in Q3 2024?"])

scores = q_emb @ doc_emb.T
```

Note that the query towers listed above are distilled from the 2B teacher and
the document towers from the 8B teacher, so they are not interchangeable across
teacher rows. Pair a document tower with the 8B-teacher query encoder that
ships with it.

---

## Training

> **Not in this repository yet.** The training and evaluation code is being
> ported from the research tree into the layout below; the interface is fixed
> but the modules have not landed. What follows is the shape it will take.

All three training entry points read the same config schema and differ only in
which tower they instantiate.

```bash
# query tower, single vector
python -m nanovdr.train.query --config configs/query/distilbert_2b.yaml

# query tower, multi vector
python -m nanovdr.train.query --config configs/query/distilbert_2b_ot.yaml

# doc tower, single vector
torchrun --nproc_per_node=2 -m nanovdr.train.doc --config configs/doc/hires_8b.yaml
```

Teacher targets are cached before training and reused across every run:

```bash
python -m nanovdr.teacher.cache --teacher qwen3-vl-embedding-8b --split train
```

<!-- TODO: state the one-off cost here. Measured for the 8B teacher over the
     1.20M-image / 1.49M-query mixture: 99.5 H200-GPU-hours total. -->

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

> **Not in this repository yet**, same as above.

`eval/` reproduces **twelve publicly released retrievers plus the teacher**
under one protocol: same evaluation driver, same input formatting, same
deduplication, same metric code. It reports retrieval quality *and* deployment
cost side by side.

```bash
python -m nanovdr.eval.vidore     --model <hf-id> --benchmarks v1 v2 v3
python -m nanovdr.eval.baselines  --all
python -m nanovdr.eval.efficiency --model <hf-id> --batch-size 8
```

Measured per model: NDCG@5 on ViDoRe v1/v2/v3, query latency, document
throughput, peak VRAM, index size per million documents, and CPU scoring
latency at 10K candidates.

The harness carries the per-baseline handling that these models need in order
to reproduce their published behaviour — float32 for BiModernVBERT, the
released chat-template path with `use_cache=False` for DSE-Qwen2, the native
flash-attention path for InternViT, and so on. If you only take one thing from
this repository, take this.

<!-- TODO: publish the raw result JSON as a dataset repo
     (nanovdr/vdr-benchmark-results) so the numbers are citable without
     rerunning anything. -->

---

## Repository layout

What has landed:

```
nanovdr/align/registry.py    the alignment-objective interface both towers share
packaging/                   the doc-tower release tooling, used for the two
                             checkpoints on the Hub:
  modeling_nanovdr_doc.py      standalone model definition (this is what
                               trust_remote_code loads from the Hub)
  build_doc_package.py         training checkpoint -> Hub-ready package
  verify_doc_package.py        loads a package and checks it against cached
                               teacher embeddings on a real ViDoRe corpus
```

Landing next: `nanovdr/towers/`, `heads.py`, `teacher/cache.py`, `tiling.py`,
`scoring.py`, the `cosine` and `ot` objectives, `train/`, `eval/`, `configs/`.

The packaging tooling is not decorative. `verify_doc_package.py` is how we
established that the released weights reproduce our internal evaluation:

```
500 pages of ViDoRe arxivqa, NanoVDR-D-HiRes
cosine(student page, teacher page)      : 0.7959 mean
NDCG@5  teacher queries x teacher pages : 86.91
NDCG@5  teacher queries x student pages : 83.07   (95.6% retention)
```

## Papers

- **NanoVDR**, single-vector query tower distilled from a 2B teacher.
  [arXiv:2603.12824](https://arxiv.org/abs/2603.12824)
- **DistilVDR**, both towers distilled from an 8B teacher, end-to-end and
  teacher-free at deployment. Preprint on arXiv; identifier being added here as
  soon as it is announced.
- **NanoVDR-v2**, multi-vector query tower. Coming soon.

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
