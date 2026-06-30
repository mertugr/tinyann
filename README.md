# tinyann

Small **C++17** in-memory vector similarity search library with a CLI.
Distance kernels use **SIMD** when available (AVX2 / SSE2 / ARM NEON, with scalar fallback).

## Features

- Fixed-dimension vectors, `int64` ids
- Metrics:
  - **cosine** — similarity (higher is better)
  - **inner_product** (ip / dot) — similarity (higher is better)
  - **euclidean** (l2) — distance (lower is better)
- **Exact** brute-force k-NN (`Index`)
- **Approximate** k-NN via **HNSW** (`HnswIndex`) — same metrics, tunable speed/recall
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

# Benchmark exact vs HNSW (synthetic unit vectors)
./build/tinyann --bench --dim 64 --n 20000 --nq 200 --k 10 \
  --metric cosine --ef 64 --M 16 --efc 200
```

Benchmark prints build/search times, QPS, **speedup vs exact**, **recall@k**, and checks HNSW result-id stability across two passes.

Vector file: `<id> <f1> … <fN>` per line (`#` comments / blanks ignored).

Binary format: magic `TANN`, version, kind (`exact` vs `hnsw`), metric, dimension, ids, vectors; HNSW also stores params, entry point, levels, adjacency lists, and RNG state.

## Ranking rules

| Metric         | Score meaning           | Order      |
|----------------|-------------------------|------------|
| Cosine         | similarity in `[-1, 1]` | descending |
| Inner product  | dot product             | descending |
| Euclidean (L2) | distance ≥ 0            | ascending  |

Equal scores tie-break by smaller `id`.

## License

MIT (or as specified by the repository owner).
