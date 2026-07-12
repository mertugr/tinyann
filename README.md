# tinyann

**Version:** `0.1.0` · **License:** MIT · **C++17** · header-only library + CLI  
**Compatibility:** [COMPATIBILITY.md](COMPATIBILITY.md) (formats, stack consumers)

Small **C++17** in-memory vector similarity search library with a CLI.
Distance kernels use **SIMD** when available (AVX2 / SSE2 / ARM NEON, with scalar fallback).

**Stack:** used by [nanorag](https://github.com/mertugr/nanorag) (≥ tinyann **0.1.0**) for dense HNSW + cosine retrieval; works standalone for exact/HNSW/IVF/IVFPQ/SQ.

## Features

- Fixed-dimension vectors, `int64` ids
- Metrics:
  - **cosine** — similarity (higher is better)
  - **inner_product** (ip / dot) — similarity (higher is better)
  - **euclidean** (l2) — distance (lower is better)
- **Exact** brute-force k-NN (`Index`)
- **Approximate** k-NN via **HNSW** (`HnswIndex`) — same metrics, tunable speed/recall
- **Approximate** k-NN via **IVF** (`IvfIndex`) — FAISS-style: train k-means (`nlist`), probe `nprobe` lists, brute-force within
- **IVF + Product Quantization (`IvfPqIndex`)** — compressed corpus: residual PQ codes (`M` bytes/vector, 8-bit subquantizers), ADC search; CLI `--index ivfpq --pq-m M`
- **Scalar quantization (SQ int8)** — `tinyann::sq::quantize` / `dequantize`, `IndexSq` stores per-vector int8 codes + scale; CLI `--sq`
- **`recall_at_k` / `HnswIndex::recall_at_k_vs`** to measure approx quality vs exact
- Edge cases: empty index, `k == 0`, `k > n`, zero vectors (cosine → score `0`)
- CLI to load vectors from a text file and run queries (`--index exact|hnsw`, optional `--recall`)
- **Binary persistence**: `Index::save` / `Index::load` and `HnswIndex::save` / `HnswIndex::load` (HNSW stores the full graph); CLI `--save` / `--load`
- **`remove(id)` / `update(id, vector)` / `contains(id)`** on both indexes; HNSW hard-deletes unlink nodes, remap indices, and reassign the entry point so the graph stays searchable
- **Filtered search** `search(query, k, predicate)` — only ids with `predicate(id) == true`; HNSW applies the filter during graph search (not post-filter of top‑k). CLI: `--allow-ids FILE`
- Unit tests (CTest), including random-data recall@10 checks, save/load byte-identical search, remove/update, and filtered search

## Why HNSW (not IVF)?

| | **HNSW** (chosen) | **IVF** |
|--|-------------------|---------|
| Insert API | Incremental `add()` — no training pass | Needs k-means training / retrain on growth |
| In-memory k-NN | Excellent recall vs latency | Needs high `nprobe` for high recall; often paired with PQ |
| Query tuning | `ef_search` at query time | `nprobe` only searches more clusters |
| Fit for tinyann | Matches “add then search” library shape | Better when you batch-build huge static corpora |

HNSW trades a modest amount of RAM (graph links) and build time for high recall without a separate clustering stage — the usual default for process-local ANN.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Optional on x86: -DTINYANN_ENABLE_AVX2=ON (default ON) for AVX2+FMA kernels
cmake --build build
ctest --test-dir build --output-on-failure
```

Active backend is reported as `distance_backend=neon|avx2|sse2|scalar` (CLI stderr / tests).

Artifacts:

- `build/tinyann` — CLI
- `build/tinyann_tests` — unit tests

## Library usage

```cpp
#include "tinyann/tinyann.hpp"

// Exact
tinyann::Index exact(/*dim=*/32, tinyann::Metric::Cosine);
exact.add(1, /* vector */);

// Approximate (HNSW)
tinyann::HnswParams p;
p.M = 16;
p.ef_construction = 200;
p.ef_search = 64;
tinyann::HnswIndex hnsw(32, tinyann::Metric::Cosine, p);
hnsw.add(1, /* vector */);

auto hits = hnsw.search(query, /*k=*/10);

// Measure quality vs exact on a query set
double rec = hnsw.recall_at_k_vs(exact, queries, /*k=*/10);
// or: tinyann::recall_at_k(approx_hits, exact_hits);

// Persistence (compact binary; HNSW includes the full graph)
exact.save("exact.tann");
auto exact2 = tinyann::Index::load("exact.tann");
hnsw.save("hnsw.tann");
auto hnsw2 = tinyann::HnswIndex::load("hnsw.tann");
// search results on loaded indexes are byte-identical to pre-save

// Remove / update (all entries with that id)
exact.remove(42);
exact.update(7, new_vector);
hnsw.remove(42);   // unlinks node, keeps graph searchable; reassigns entry if needed
hnsw.update(7, new_vector);

// Filtered search (eligible ids only; HNSW keeps exploring under the filter)
auto hits = index.search(query, 10, [](std::int64_t id) { return id % 2 == 0; });
auto ahits = hnsw.search(query, 10, /*ef=*/64, [](std::int64_t id) { return id > 0; });

// Scalar quantization (int8, symmetric per-vector scale)
std::vector<std::int8_t> codes;
float scale = 0.f;
tinyann::sq::quantize(vec, codes, scale);
auto approx = tinyann::sq::dequantize(codes, scale);

tinyann::IndexSq sq_index(dim, tinyann::Metric::Cosine);
sq_index.add(1, vec);
auto shits = sq_index.search(query, 10);

// IVF (train then add — FAISS-like, full-float lists)
tinyann::IvfParams ip;
ip.nlist = 100;
ip.nprobe = 10;
tinyann::IvfIndex ivf(dim, tinyann::Metric::Cosine, ip);
ivf.train(training_vectors);
ivf.add(1, vec);
auto ihits = ivf.search(query, 10);

// IVFPQ — large corpora in less RAM (residual product quantization)
tinyann::IvfPqParams pp;
pp.nlist = 100;
pp.nprobe = 10;
pp.M = 8;  // dim must be divisible by M; stores M bytes per vector
pp.use_opq = true;   // optional: learned rotation before residual PQ
pp.opq_iters = 10;
tinyann::IvfPqIndex ivfpq(dim, tinyann::Metric::Cosine, pp);
ivfpq.train(training_vectors);
ivfpq.add(1, vec);
auto phits = ivfpq.search(query, 10);
ivfpq.save("corpus.ivfpq.tann");
auto loaded = tinyann::IvfPqIndex::load("corpus.ivfpq.tann");
```

Header-only: link the `tinyann` CMake interface target (or add `include/`).

## CLI

```bash
# Exact
./build/tinyann --dim 3 --metric cosine \
  --vectors data/vectors.txt --query data/query.txt --k 3 --index exact

# HNSW + print mean recall@k against exact on the same queries
./build/tinyann --dim 3 --metric cosine \
  --vectors data/vectors.txt --query data/query.txt --k 3 \
  --index hnsw --ef 64 --M 16 --recall

# Save / load binary index
./build/tinyann --dim 3 --metric cosine --vectors data/vectors.txt \
  --index exact --save /tmp/exact.tann
./build/tinyann --load /tmp/exact.tann --index exact \
  --query data/query.txt --k 3

./build/tinyann --dim 3 --metric cosine --vectors data/vectors.txt \
  --index hnsw --save /tmp/hnsw.tann
./build/tinyann --load /tmp/hnsw.tann --index hnsw \
  --query data/query.txt --k 3

# Filtered search: only ids listed in allow file (one int64 per line)
./build/tinyann --dim 3 --metric cosine --vectors data/vectors.txt \
  --query data/query.txt --k 3 --index hnsw --allow-ids data/allow_ids.txt --recall

# Exact with int8 scalar quantization
./build/tinyann --dim 3 --metric cosine --vectors data/vectors.txt \
  --query data/query.txt --k 3 --index exact --sq --recall

# IVF
./build/tinyann --dim 3 --metric cosine --vectors data/vectors.txt \
  --query data/query.txt --k 3 --index ivf --nlist 4 --nprobe 2 --recall

# IVFPQ (dim must be divisible by --pq-m; sample data dim=3 needs M=1 or use higher-dim data)
./build/tinyann --dim 64 --metric cosine --vectors my_vectors.txt \
  --query my_query.txt --k 10 --index ivfpq --nlist 100 --nprobe 10 --pq-m 8 --recall

# IVFPQ + OPQ (optimized product quantization)
./build/tinyann --dim 64 --metric cosine --vectors my_vectors.txt \
  --query my_query.txt --k 10 --index ivfpq --pq-m 8 --opq --opq-iters 10 --recall

# Benchmark exact vs HNSW vs IVF (synthetic unit vectors)
./build/tinyann --bench --dim 64 --n 20000 --nq 200 --k 10 \
  --metric cosine --ef 64 --M 16 --efc 200 --nlist 100 --nprobe 10
```

Benchmark prints build/search times, QPS, **speedup vs exact**, **recall@k** for HNSW and IVF, and id stability checks.

Vector file: optional integer `<id>` then `<f1> … <fN>` per line (`#` comments / blanks ignored). The id is parsed as **int64 text** (not float); lines with only `N` floats get auto-assigned sequential ids.

Binary format: magic `TANN`, version, kind (`exact` / `hnsw` / `ivf` / `sq` / `ivfpq`), metric, dimension, ids, vectors (or PQ codes for IVFPQ); HNSW also stores params, entry point, levels, adjacency lists, and RNG state. **Host-endian only** — files are not portable across different-endian machines (no endian marker in the header).

**IVFPQ notes:** residual product quantization (encode `x - coarse_centroid`). **OPQ** (`use_opq`): learn orthogonal `R`, PQ on `R r`, reconstruct with `Rᵀ`. **Euclidean:** residual L2 ADC in PQ space. **Inner product:** `IP(q,c)+IP(Rq, decode)`. **Cosine:** normalize for PQ path; scores use `cosine_similarity(query, reconstruct(code))`. Best for large static-ish corpora; train once, then `add` / `search` / `save` / `load`.

### API notes (from design review)

- **Finite vectors:** `add` / `update` / `search` reject NaN/Inf components.
- **HNSW concurrency:** concurrent `search` on the **same** `HnswIndex` is safe (per-call visit/query scratch). Do not interleave `search` with concurrent `add`/`remove`/`update` (no writer locks).
- **HNSW / IVF capacity:** internal node indices are `int`; `add` fails if `size() >= INT_MAX`.
- **`add` exception safety:** basic guarantee only (a throw while growing storage may leave partial state).
- **Filtered HNSW:** not a post-filter of unfiltered top‑k; very selective filters may miss some eligibles once the eligible heap is full (raise `ef` or use exact search for perfect filtered recall).

## Ranking rules

| Metric         | Score meaning           | Order      |
|----------------|-------------------------|------------|
| Cosine         | similarity in `[-1, 1]` | descending |
| Inner product  | dot product             | descending |
| Euclidean (L2) | distance ≥ 0            | ascending  |

Equal scores tie-break by smaller `id`.

## License

MIT — see [LICENSE](LICENSE).

Binary formats, package consumers, and stack matrix: [COMPATIBILITY.md](COMPATIBILITY.md).
