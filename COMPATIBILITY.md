# Compatibility

**Library version:** `0.1.0` (see [VERSION](VERSION), `CMakeLists.txt` `project(... VERSION ...)`).

tinyann is a leaf dependency in the local stack:

```
tinyann 0.1.x  ──►  nanorag 0.1.x
nanollm 0.4.x  ──►  nanorag 0.1.x
```

| Consumer | Package requirement | Role |
|----------|---------------------|------|
| **nanorag** | `find_package(tinyann 0.1)` / submodule | Dense ANN (HNSW + cosine) for retrieval |
| Standalone | — | CLI + CTest |

## Formats

| Artifact | Magic | Version | Notes |
|----------|-------|---------|--------|
| Index file (`*.tann`) | `TANN` | 1 | kinds: exact=1, hnsw=2, ivf=3, sq=4, ivfpq=5 |
| Host endian | — | — | **No endian marker** — not portable across big/little endian hosts |

## API surface used by nanorag

- `tinyann::HnswIndex` (cosine metric)
- `add` / `search` / `save` / `load`
- Fixed-dim `float` vectors, `int64` ids

IVF / IVFPQ / SQ are supported by the library and CLI but are **not** required by nanorag today.

## CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build --output-on-failure
cmake --install build --prefix $PREFIX   # exports tinyann::tinyann
```

Options: `TINYANN_BUILD_CLI`, `TINYANN_BUILD_TESTS`, `TINYANN_INSTALL`, `TINYANN_ENABLE_AVX2` (x86).

## Related docs

- [nanorag COMPATIBILITY.md](https://github.com/mertugr/nanorag/blob/main/COMPATIBILITY.md) — full stack matrix
- [nanollm COMPATIBILITY.md](https://github.com/mertugr/nanollm/blob/main/COMPATIBILITY.md)

## License

MIT — see [LICENSE](LICENSE).
