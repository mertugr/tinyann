# tinyann

Small **C++17** in-memory vector similarity search library with a CLI.

## Features

- Fixed-dimension vectors, `int64` ids
- Metrics:
  - **cosine** — similarity (higher is better)
  - **inner_product** (ip / dot) — similarity (higher is better)
  - **euclidean** (l2) — distance (lower is better)
- Exact brute-force **k-NN** (`search(query, k)`)
- Edge cases: empty index, `k == 0`, `k > n`, zero vectors (cosine → score `0`)
- CLI to load vectors from a text file and run queries
- Unit tests (CTest)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Artifacts:

- `build/tinyann` — CLI
- `build/tinyann_tests` — unit tests

## Library usage

```cpp
#include "tinyann/tinyann.hpp"

tinyann::Index index(/*dim=*/3, tinyann::Metric::Cosine);
index.add(1, {1.f, 0.f, 0.f});
index.add(2, {0.f, 1.f, 0.f});

auto hits = index.search({1.f, 0.f, 0.f}, /*k=*/2);
// hits[i].id, hits[i].score — best first
```

Header-only: link against the `tinyann` CMake interface target (or add `include/` to your include path).

## CLI

Vector file format — one vector per line:

```text
<id> <f1> <f2> ... <fN>
```

Lines starting with `#` and blank lines are ignored. If the leading id is omitted, sequential ids starting at `0` are assigned.

Query file — one query per line, either `f1 … fN` or `id f1 … fN` (id ignored).

```bash
./build/tinyann \
  --dim 3 \
  --metric cosine \
  --vectors data/vectors.txt \
  --query data/query.txt \
  --k 3
```

Output: `rank<TAB>id<TAB>score` per hit.

Metrics accepted by `--metric`: `cosine`, `cos`, `euclidean`, `l2`, `inner_product`, `ip`, `dot`.

## Ranking rules

| Metric         | Score meaning              | Order        |
|----------------|----------------------------|--------------|
| Cosine         | similarity in `[-1, 1]`    | descending   |
| Inner product  | dot product                | descending   |
| Euclidean (L2) | distance ≥ 0               | ascending    |

Equal scores are tie-broken by smaller `id` first (deterministic).

## License

MIT (or as specified by the repository owner).
