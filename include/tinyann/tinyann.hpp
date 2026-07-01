#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#if defined(__FMA__)
#include <immintrin.h>
#endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tinyann {

/// Predicate for filtered search: return true if `id` is eligible.
using IdPredicate = std::function<bool(std::int64_t)>;

/// Distance / similarity metrics supported by the indexes.
enum class Metric {
    Cosine,        ///< Cosine similarity in [-1, 1]; higher is better
    Euclidean,     ///< L2 distance; lower is better
    InnerProduct   ///< Dot product; higher is better
};

/// A single search hit: vector id and metric score.
struct SearchResult {
    std::int64_t id;
    float score;
};

// ---------------------------------------------------------------------------
// SIMD distance kernels (compile-time backend selection)
// Priority: AVX2 → SSE2 → ARM NEON → scalar
// ---------------------------------------------------------------------------

namespace simd {

#if defined(__AVX2__)

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    __m256 vacc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    float sum = hsum256(vacc);
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float squared_l2(const float* a, const float* b, std::size_t dim) {
    __m256 vacc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 vd = _mm256_sub_ps(va, vb);
        vacc = _mm256_fmadd_ps(vd, vd, vacc);
    }
    float sum = hsum256(vacc);
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline void cosine_parts(const float* a, const float* b, std::size_t dim, float& dot, float& na,
                         float& nb) {
    __m256 vdot = _mm256_setzero_ps();
    __m256 vna = _mm256_setzero_ps();
    __m256 vnb = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna = _mm256_fmadd_ps(va, va, vna);
        vnb = _mm256_fmadd_ps(vb, vb, vnb);
    }
    dot = hsum256(vdot);
    na = hsum256(vna);
    nb = hsum256(vnb);
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
}

inline const char* backend_name() { return "avx2"; }

#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)

// Pure SSE2 horizontal sum of 4 floats.
inline float hsum128_sse2(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    __m128 vacc = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
#if defined(__FMA__)
        vacc = _mm_fmadd_ps(va, vb, vacc);
#else
        vacc = _mm_add_ps(vacc, _mm_mul_ps(va, vb));
#endif
    }
    float sum = hsum128_sse2(vacc);
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float squared_l2(const float* a, const float* b, std::size_t dim) {
    __m128 vacc = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
        const __m128 vd = _mm_sub_ps(va, vb);
#if defined(__FMA__)
        vacc = _mm_fmadd_ps(vd, vd, vacc);
#else
        vacc = _mm_add_ps(vacc, _mm_mul_ps(vd, vd));
#endif
    }
    float sum = hsum128_sse2(vacc);
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline void cosine_parts(const float* a, const float* b, std::size_t dim, float& dot, float& na,
                         float& nb) {
    __m128 vdot = _mm_setzero_ps();
    __m128 vna = _mm_setzero_ps();
    __m128 vnb = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
#if defined(__FMA__)
        vdot = _mm_fmadd_ps(va, vb, vdot);
        vna = _mm_fmadd_ps(va, va, vna);
        vnb = _mm_fmadd_ps(vb, vb, vnb);
#else
        vdot = _mm_add_ps(vdot, _mm_mul_ps(va, vb));
        vna = _mm_add_ps(vna, _mm_mul_ps(va, va));
        vnb = _mm_add_ps(vnb, _mm_mul_ps(vb, vb));
#endif
    }
    dot = hsum128_sse2(vdot);
    na = hsum128_sse2(vna);
    nb = hsum128_sse2(vnb);
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
}

inline const char* backend_name() { return "sse2"; }

#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

inline float hsum128_neon(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
#endif
}

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    float32x4_t vacc = vdupq_n_f32(0.f);
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vacc = vmlaq_f32(vacc, va, vb);
    }
    float sum = hsum128_neon(vacc);
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float squared_l2(const float* a, const float* b, std::size_t dim) {
    float32x4_t vacc = vdupq_n_f32(0.f);
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        const float32x4_t vd = vsubq_f32(va, vb);
        vacc = vmlaq_f32(vacc, vd, vd);
    }
    float sum = hsum128_neon(vacc);
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline void cosine_parts(const float* a, const float* b, std::size_t dim, float& dot, float& na,
                         float& nb) {
    float32x4_t vdot = vdupq_n_f32(0.f);
    float32x4_t vna = vdupq_n_f32(0.f);
    float32x4_t vnb = vdupq_n_f32(0.f);
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vdot = vmlaq_f32(vdot, va, vb);
        vna = vmlaq_f32(vna, va, va);
        vnb = vmlaq_f32(vnb, vb, vb);
    }
    dot = hsum128_neon(vdot);
    na = hsum128_neon(vna);
    nb = hsum128_neon(vnb);
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
}

inline const char* backend_name() { return "neon"; }

#else

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    float sum = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float squared_l2(const float* a, const float* b, std::size_t dim) {
    float sum = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline void cosine_parts(const float* a, const float* b, std::size_t dim, float& dot, float& na,
                         float& nb) {
    dot = 0.f;
    na = 0.f;
    nb = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
}

inline const char* backend_name() { return "scalar"; }

#endif

}  // namespace simd

// ---------------------------------------------------------------------------
// Shared metric helpers (used by exact Index and HnswIndex)
// ---------------------------------------------------------------------------

inline bool higher_is_better(Metric m) noexcept { return m != Metric::Euclidean; }

/// Active SIMD/scalar backend name ("avx2", "sse2", "neon", or "scalar").
inline const char* distance_backend() { return simd::backend_name(); }

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    return simd::inner_product(a, b, dim);
}

inline float euclidean_distance(const float* a, const float* b, std::size_t dim) {
    return std::sqrt(simd::squared_l2(a, b, dim));
}

/// Cosine similarity. Zero-norm on either side yields 0 (safe default).
inline float cosine_similarity(const float* a, const float* b, std::size_t dim) {
    float dot = 0.f;
    float na = 0.f;
    float nb = 0.f;
    simd::cosine_parts(a, b, dim, dot, na, nb);
    if (na == 0.f || nb == 0.f) {
        return 0.f;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

/// Require every component to be finite (not NaN/Inf). Used by add/update/search.
inline void require_finite(const float* v, std::size_t dim, const char* what) {
    for (std::size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(v[i])) {
            throw std::invalid_argument(std::string(what) + ": non-finite component at index " +
                                        std::to_string(i));
        }
    }
}

inline void require_finite(const std::vector<float>& v, const char* what) {
    require_finite(v.data(), v.size(), what);
}

/// Metric score as exposed to users (similarity or L2 distance).
inline float metric_score(Metric m, const float* a, const float* b, std::size_t dim) {
    switch (m) {
        case Metric::Cosine:
            return cosine_similarity(a, b, dim);
        case Metric::Euclidean:
            return euclidean_distance(a, b, dim);
        case Metric::InnerProduct:
            return inner_product(a, b, dim);
    }
    return 0.f;
}

/// Internal "distance" for ANN graph navigation: always smaller = closer.
/// Cosine → 1 - sim; InnerProduct → -dot; Euclidean → **squared** L2 (order-preserving, no sqrt).
inline float ann_distance(Metric m, const float* a, const float* b, std::size_t dim) {
    switch (m) {
        case Metric::Cosine:
            return 1.f - cosine_similarity(a, b, dim);
        case Metric::Euclidean:
            return simd::squared_l2(a, b, dim);
        case Metric::InnerProduct:
            return -inner_product(a, b, dim);
    }
    return 0.f;
}

/// L2 norm of a vector (for cosine caches).
inline float l2_norm(const float* v, std::size_t dim) {
    return std::sqrt(inner_product(v, v, dim));
}

inline const char* metric_name(Metric m) {
    switch (m) {
        case Metric::Cosine:
            return "cosine";
        case Metric::Euclidean:
            return "euclidean";
        case Metric::InnerProduct:
            return "inner_product";
    }
    return "unknown";
}

inline Metric parse_metric(const std::string& name) {
    if (name == "cosine" || name == "cos") {
        return Metric::Cosine;
    }
    if (name == "euclidean" || name == "l2" || name == "euclid") {
        return Metric::Euclidean;
    }
    if (name == "inner_product" || name == "ip" || name == "dot") {
        return Metric::InnerProduct;
    }
    throw std::invalid_argument("unknown metric: " + name);
}

/// Rank SearchResults best-first for the given metric (tie-break: smaller id).
inline void rank_results(std::vector<SearchResult>& results, Metric metric) {
    const bool hib = higher_is_better(metric);
    std::sort(results.begin(), results.end(), [hib](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) {
            return hib ? (a.score > b.score) : (a.score < b.score);
        }
        return a.id < b.id;
    });
}

/// Recall@k of approximate results vs exact ground truth.
inline double recall_at_k(const std::vector<SearchResult>& approx,
                          const std::vector<SearchResult>& exact) {
    if (exact.empty()) {
        return 1.0;
    }
    std::unordered_set<std::int64_t> truth;
    truth.reserve(exact.size() * 2);
    for (const auto& r : exact) {
        truth.insert(r.id);
    }
    std::size_t hits = 0;
    std::unordered_set<std::int64_t> seen;
    for (const auto& r : approx) {
        if (seen.insert(r.id).second && truth.count(r.id)) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(truth.size());
}

/// Mean recall@k over multiple (approx, exact) pairs.
inline double mean_recall_at_k(
    const std::vector<std::vector<SearchResult>>& approx_batch,
    const std::vector<std::vector<SearchResult>>& exact_batch) {
    if (approx_batch.size() != exact_batch.size()) {
        throw std::invalid_argument("mean_recall_at_k: batch size mismatch");
    }
    if (approx_batch.empty()) {
        return 1.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < approx_batch.size(); ++i) {
        sum += recall_at_k(approx_batch[i], exact_batch[i]);
    }
    return sum / static_cast<double>(approx_batch.size());
}

// ---------------------------------------------------------------------------
// Binary persistence helpers
// Format: magic "TANN" | version u32 | kind u32 | ... payload ...
// kind: 1 = Exact Index, 2 = HnswIndex, 3 = IvfIndex, 4 = IndexSq
//
// Endianness: multi-byte fields are written as raw host memory (native endian).
// Files are portable only across machines with the **same endianness** (and the
// same float format). There is no endian marker in the header — do not load a
// file produced on a different-endian host.
// ---------------------------------------------------------------------------

namespace detail {

constexpr char kMagic[4] = {'T', 'A', 'N', 'N'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kKindExact = 1;
constexpr std::uint32_t kKindHnsw = 2;
constexpr std::uint32_t kKindIvf = 3;
constexpr std::uint32_t kKindSq = 4;     // exact index with int8 scalar quantization
constexpr std::uint32_t kKindIvfPq = 5;  // IVF + product quantization (compressed codes)

inline void write_bytes(std::ostream& out, const void* data, std::size_t n) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!out) {
        throw std::runtime_error("tinyann: failed writing to stream");
    }
}

inline void read_bytes(std::istream& in, void* data, std::size_t n) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n));
    if (!in || static_cast<std::size_t>(in.gcount()) != n) {
        throw std::runtime_error("tinyann: failed reading from stream (truncated or corrupt)");
    }
}

/// Multiply `a * b` for allocation/read sizes; throws on overflow (hostile/corrupt files).
inline std::size_t checked_mul_size(std::size_t a, std::size_t b, const char* what) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::runtime_error(std::string("tinyann: size overflow (") + what + ")");
    }
    return a * b;
}

/// Convert a file `uint64` size field to `size_t`, rejecting values that do not fit.
inline std::size_t size_from_u64(std::uint64_t v, const char* what) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("tinyann: size does not fit platform (") + what + ")");
    }
    return static_cast<std::size_t>(v);
}

template <typename T>
inline void write_pod(std::ostream& out, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD required");
    write_bytes(out, &v, sizeof(T));
}

template <typename T>
inline T read_pod(std::istream& in) {
    static_assert(std::is_trivially_copyable<T>::value, "POD required");
    T v{};
    read_bytes(in, &v, sizeof(T));
    return v;
}

inline void write_header(std::ostream& out, std::uint32_t kind) {
    write_bytes(out, kMagic, 4);
    write_pod(out, kFormatVersion);
    write_pod(out, kind);
}

/// Read magic+version+kind; returns kind. Throws on mismatch.
inline std::uint32_t read_header(std::istream& in, std::uint32_t expected_kind) {
    char magic[4];
    read_bytes(in, magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0) {
        throw std::runtime_error("tinyann: invalid file magic (not a tinyann index)");
    }
    const auto ver = read_pod<std::uint32_t>(in);
    if (ver != kFormatVersion) {
        throw std::runtime_error("tinyann: unsupported format version " + std::to_string(ver));
    }
    const auto kind = read_pod<std::uint32_t>(in);
    if (kind != expected_kind) {
        throw std::runtime_error("tinyann: index kind mismatch (file kind=" + std::to_string(kind) +
                                 ", expected " + std::to_string(expected_kind) + ")");
    }
    return kind;
}

inline Metric metric_from_u32(std::uint32_t m) {
    if (m > static_cast<std::uint32_t>(Metric::InnerProduct)) {
        throw std::runtime_error("tinyann: invalid metric id in file");
    }
    return static_cast<Metric>(m);
}

inline void write_ids_and_vectors(std::ostream& out, std::size_t dimension, Metric metric,
                                  const std::vector<std::int64_t>& ids,
                                  const std::vector<float>& data) {
    write_pod(out, static_cast<std::uint32_t>(metric));
    write_pod(out, static_cast<std::uint64_t>(dimension));
    write_pod(out, static_cast<std::uint64_t>(ids.size()));
    if (!ids.empty()) {
        write_bytes(out, ids.data(), ids.size() * sizeof(std::int64_t));
    }
    if (!data.empty()) {
        write_bytes(out, data.data(), data.size() * sizeof(float));
    }
}

inline void read_ids_and_vectors(std::istream& in, std::size_t& dimension, Metric& metric,
                                 std::vector<std::int64_t>& ids, std::vector<float>& data) {
    metric = metric_from_u32(read_pod<std::uint32_t>(in));
    dimension = size_from_u64(read_pod<std::uint64_t>(in), "dimension");
    const std::size_t count = size_from_u64(read_pod<std::uint64_t>(in), "vector count");
    if (dimension == 0) {
        throw std::runtime_error("tinyann: corrupt file (dimension == 0)");
    }
    const std::size_t n_floats = checked_mul_size(count, dimension, "count * dimension");
    const std::size_t ids_bytes =
        checked_mul_size(count, sizeof(std::int64_t), "count * sizeof(int64)");
    const std::size_t data_bytes =
        checked_mul_size(n_floats, sizeof(float), "count * dimension * sizeof(float)");
    ids.resize(count);
    data.resize(n_floats);
    if (count > 0) {
        read_bytes(in, ids.data(), ids_bytes);
        read_bytes(in, data.data(), data_bytes);
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Exact (brute-force) index
// ---------------------------------------------------------------------------

/// In-memory exact (brute-force) vector similarity index.
///
/// Vectors must be finite (no NaN/Inf). `add` provides the **basic** exception
/// guarantee only (a throw while growing storage may leave containers partially
/// updated); it is not strongly exception-safe.
class Index {
public:
    explicit Index(std::size_t dimension, Metric metric = Metric::Cosine)
        : dimension_(dimension), metric_(metric) {
        if (dimension_ == 0) {
            throw std::invalid_argument("tinyann::Index: dimension must be > 0");
        }
    }

    Index(const Index&) = default;
    Index(Index&&) noexcept = default;
    Index& operator=(const Index&) = default;
    Index& operator=(Index&&) noexcept = default;
    ~Index() = default;

    std::size_t dimension() const noexcept { return dimension_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return ids_.size(); }
    bool empty() const noexcept { return ids_.empty(); }

    /// Append one vector. Basic exception safety only (see class note).
    void add(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::Index::add");
        ids_.push_back(id);
        data_.insert(data_.end(), vector.begin(), vector.end());
    }

    /// Remove every entry with the given id.
    /// @return true if at least one vector was removed.
    bool remove(std::int64_t id) {
        bool removed = false;
        for (std::size_t i = 0; i < ids_.size();) {
            if (ids_[i] != id) {
                ++i;
                continue;
            }
            const std::size_t last = ids_.size() - 1;
            if (i != last) {
                ids_[i] = ids_[last];
                std::copy(data_.begin() + static_cast<std::ptrdiff_t>(last * dimension_),
                          data_.begin() + static_cast<std::ptrdiff_t>((last + 1) * dimension_),
                          data_.begin() + static_cast<std::ptrdiff_t>(i * dimension_));
            }
            ids_.pop_back();
            data_.resize(ids_.size() * dimension_);
            removed = true;
            // re-check index i (now holds former last, or we shrunk past it)
        }
        return removed;
    }

    /// Replace the vector for every entry with the given id (in place).
    /// @return true if at least one entry was updated.
    /// @throws std::invalid_argument on dimension mismatch or non-finite components
    bool update(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::update: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::Index::update");
        bool updated = false;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (ids_[i] != id) {
                continue;
            }
            std::copy(vector.begin(), vector.end(),
                      data_.begin() + static_cast<std::ptrdiff_t>(i * dimension_));
            updated = true;
        }
        return updated;
    }

    /// True if at least one entry has this id.
    bool contains(std::int64_t id) const {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    /// Exact k-NN search. Best-first; empty if k==0 or index empty; all if k>n.
    /// Query must be finite.
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        return search(query, k, [](std::int64_t) { return true; });
    }

    /// Exact filtered k-NN: only ids for which `predicate(id)` is true are eligible.
    /// Returns up to k nearest eligible hits (may be fewer / empty). Ranked best-first.
    /// A predicate that accepts everything matches unfiltered `search(query, k)`.
    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        if (query.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::search: query dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(query.size()) + ")");
        }
        require_finite(query, "tinyann::Index::search");
        if (k == 0 || empty()) {
            return {};
        }

        std::vector<SearchResult> eligible;
        eligible.reserve(std::min(k, size()));
        const float* base = data_.data();
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (!predicate(ids_[i])) {
                continue;
            }
            eligible.push_back(SearchResult{
                ids_[i], metric_score(metric_, query.data(), base + i * dimension_, dimension_)});
        }
        if (eligible.empty()) {
            return {};
        }

        const std::size_t take = std::min(k, eligible.size());
        const bool hib = higher_is_better(metric_);
        auto better = [hib](const SearchResult& a, const SearchResult& b) {
            if (a.score != b.score) {
                return hib ? (a.score > b.score) : (a.score < b.score);
            }
            return a.id < b.id;
        };

        if (take < eligible.size()) {
            std::partial_sort(eligible.begin(),
                              eligible.begin() + static_cast<std::ptrdiff_t>(take), eligible.end(),
                              better);
            eligible.resize(take);
        } else {
            std::sort(eligible.begin(), eligible.end(), better);
        }
        return eligible;
    }

    float score(const std::vector<float>& a, const std::vector<float>& b) const {
        if (a.size() != dimension_ || b.size() != dimension_) {
            throw std::invalid_argument("tinyann::Index::score: dimension mismatch");
        }
        return metric_score(metric_, a.data(), b.data(), dimension_);
    }

    /// Save compact binary index to path (creates/overwrites file).
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("tinyann::Index::save: cannot open " + path);
        }
        detail::write_header(out, detail::kKindExact);
        detail::write_ids_and_vectors(out, dimension_, metric_, ids_, data_);
    }

    /// Load exact index from a file written by Index::save.
    static Index load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("tinyann::Index::load: cannot open " + path);
        }
        detail::read_header(in, detail::kKindExact);
        std::size_t dimension = 0;
        Metric metric = Metric::Cosine;
        std::vector<std::int64_t> ids;
        std::vector<float> data;
        detail::read_ids_and_vectors(in, dimension, metric, ids, data);
        Index idx(dimension, metric);
        idx.ids_ = std::move(ids);
        idx.data_ = std::move(data);
        return idx;
    }

    static const char* metric_name(Metric m) { return tinyann::metric_name(m); }
    static Metric parse_metric(const std::string& name) { return tinyann::parse_metric(name); }

    const std::vector<std::int64_t>& ids() const noexcept { return ids_; }
    const std::vector<float>& data() const noexcept { return data_; }

private:
    std::size_t dimension_;
    Metric metric_;
    std::vector<std::int64_t> ids_;
    std::vector<float> data_;
};

// ---------------------------------------------------------------------------
// Approximate index: HNSW
// ---------------------------------------------------------------------------

/// Parameters for HNSW graph construction and search.
struct HnswParams {
    /// Max bidirectional links per node on layers > 0 (layer 0 uses 2*M).
    std::size_t M = 16;
    /// Width of the candidate list while inserting nodes.
    std::size_t ef_construction = 200;
    /// Default width of the candidate list while searching (overridable per query).
    std::size_t ef_search = 64;
    /// RNG seed for level assignment (reproducible builds in tests).
    std::uint64_t seed = 42;
};

/// In-memory approximate k-NN index based on HNSW.
///
/// **Concurrency:** concurrent `search` (including filtered overloads) on the
/// *same* instance is safe — visit stamps and query-norm live in per-call
/// scratch, not shared mutable fields. Concurrent `add` / `remove` / `update`
/// with each other or with `search` is **not** supported (no writer locks).
///
/// **Capacity:** node indices are `int`; `add` rejects when `size() >= INT_MAX`.
///
/// **Exception safety:** `add` has the basic guarantee only (partial node on throw).
///
/// **Filtered search:** ineligible nodes may still be traversed for connectivity,
/// but expansion is gated by the eligible heap — very selective filters can miss
/// some eligible ids (not a post-filter of unfiltered top-k; see search docs).
class HnswIndex {
public:
    explicit HnswIndex(std::size_t dimension, Metric metric = Metric::Cosine,
                       HnswParams params = {})
        : dimension_(dimension),
          metric_(metric),
          params_(params),
          rng_(params.seed),
          level_mult_(1.0 / std::log(std::max<std::size_t>(params.M, 2))) {
        validate_params(dimension_, params_);
    }

    std::size_t dimension() const noexcept { return dimension_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return ids_.size(); }
    bool empty() const noexcept { return ids_.empty(); }
    const HnswParams& params() const noexcept { return params_; }

    void set_ef_search(std::size_t ef) {
        if (ef == 0) {
            throw std::invalid_argument("ef_search must be > 0");
        }
        params_.ef_search = ef;
    }

    /// Insert one node. Basic exception safety only. Rejects non-finite vectors and
    /// when the graph would exceed `INT_MAX` nodes (internal index type limit).
    void add(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::HnswIndex::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::HnswIndex::add");
        if (ids_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "tinyann::HnswIndex::add: maximum number of nodes (INT_MAX) reached");
        }

        const int node = static_cast<int>(ids_.size());
        ids_.push_back(id);
        data_.insert(data_.end(), vector.begin(), vector.end());
        norms_.push_back(l2_norm(vector.data(), dimension_));

        const int level = random_level();
        levels_.push_back(level);
        neighbors_.emplace_back();
        neighbors_.back().resize(static_cast<std::size_t>(level) + 1);

        if (node == 0) {
            entry_point_ = 0;
            max_level_ = level;
            return;
        }

        int curr = entry_point_;
        for (int lc = max_level_; lc > level; --lc) {
            curr = greedy_update(node, curr, lc);
        }

        for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
            auto candidates =
                search_layer(node, /*query_is_node=*/true, curr, params_.ef_construction, lc);
            const std::size_t M_max = (lc == 0) ? (params_.M * 2) : params_.M;
            auto selected = select_neighbors(candidates, M_max);

            neighbors_[static_cast<std::size_t>(node)][static_cast<std::size_t>(lc)] = selected;

            for (int nb : selected) {
                auto& nb_links =
                    neighbors_[static_cast<std::size_t>(nb)][static_cast<std::size_t>(lc)];
                nb_links.push_back(node);
                if (nb_links.size() > M_max) {
                    std::vector<std::pair<float, int>> scored;
                    scored.reserve(nb_links.size());
                    for (int x : nb_links) {
                        scored.emplace_back(distance_nodes(nb, x), x);
                    }
                    std::sort(scored.begin(), scored.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    nb_links.clear();
                    for (std::size_t i = 0; i < M_max && i < scored.size(); ++i) {
                        nb_links.push_back(scored[i].second);
                    }
                }
            }

            if (!candidates.empty()) {
                curr = candidates[0].second;
            }
        }

        if (level > max_level_) {
            entry_point_ = node;
            max_level_ = level;
        }
    }

    /// Remove every node with the given id. Unlinks the node from the HNSW graph,
    /// swap-removes storage, and remaps neighbor indices so the graph stays
    /// searchable. If the entry point is removed, reassigns it to a remaining
    /// node with maximum level (smallest index on ties).
    /// @return true if at least one node was removed.
    bool remove(std::int64_t id) {
        bool removed = false;
        for (int i = static_cast<int>(ids_.size()) - 1; i >= 0; --i) {
            if (ids_[static_cast<std::size_t>(i)] == id) {
                remove_node_at(i);
                removed = true;
            }
        }
        return removed;
    }

    /// Replace the vector for every node with the given id (in place; graph
    /// topology unchanged). Search remains valid; links may be slightly
    /// suboptimal relative to the new embedding.
    /// @return true if at least one node was updated.
    bool update(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::HnswIndex::update: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::HnswIndex::update");
        bool updated = false;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (ids_[i] != id) {
                continue;
            }
            std::copy(vector.begin(), vector.end(),
                      data_.begin() + static_cast<std::ptrdiff_t>(i * dimension_));
            norms_[i] = l2_norm(vector.data(), dimension_);
            updated = true;
        }
        return updated;
    }

    /// True if at least one node has this id.
    bool contains(std::int64_t id) const {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    /// Approximate k-NN. Concurrent-safe with other `search` calls on this instance
    /// (see class note); not concurrent with mutating methods.
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        return search(query, k, params_.ef_search, [](std::int64_t) { return true; });
    }

    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k,
                                     std::size_t ef) const {
        return search(query, k, ef, [](std::int64_t) { return true; });
    }

    /// Filtered approximate k-NN with default `ef_search`.
    /// `predicate(id)` selects eligible ids. Navigation still uses the full graph
    /// (ineligible nodes can be traversed); only eligible nodes enter the result
    /// candidate set — not a post-filter of an unfiltered top-k.
    ///
    /// **Tradeoff:** once the eligible candidate heap is full, nodes farther than
    /// the worst eligible (including ineligible "bridge" nodes) are not expanded.
    /// Very selective filters may therefore miss some eligible ids; raise `ef` or
    /// use exact `Index` when perfect filtered recall is required.
    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        return search(query, k, params_.ef_search, std::move(predicate));
    }

    /// Filtered approximate k-NN with explicit ef (exploration width over eligible hits).
    /// See overload without `ef` for the selective-filter tradeoff.
    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, std::size_t ef,
                Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        if (query.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::HnswIndex::search: query dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(query.size()) + ")");
        }
        require_finite(query, "tinyann::HnswIndex::search");
        if (k == 0 || empty()) {
            return {};
        }
        if (ef == 0) {
            throw std::invalid_argument("ef must be > 0");
        }
        ef = std::max(ef, k);

        // Per-call scratch: concurrent search on one instance must not share visit/query state.
        SearchScratch scratch;
        if (metric_ == Metric::Cosine) {
            scratch.query_norm = l2_norm(query.data(), dimension_);
        }

        int curr = entry_point_;
        for (int lc = max_level_; lc > 0; --lc) {
            // Upper layers: navigate on the full graph (no filter) for connectivity.
            curr = greedy_update_query(query.data(), curr, lc, scratch);
        }

        auto candidates = search_layer_query_filtered(query.data(), curr, ef, /*layer=*/0,
                                                      scratch, predicate);
        const std::size_t take = std::min(k, candidates.size());

        std::vector<SearchResult> out;
        out.reserve(take);
        for (std::size_t i = 0; i < take; ++i) {
            const int node = candidates[i].second;
            const float* vec = data_.data() + static_cast<std::size_t>(node) * dimension_;
            out.push_back(SearchResult{
                ids_[static_cast<std::size_t>(node)],
                metric_score(metric_, query.data(), vec, dimension_)});
        }
        rank_results(out, metric_);
        return out;
    }

    double recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                          std::size_t k, std::size_t ef = 0) const {
        if (ef == 0) {
            ef = params_.ef_search;
        }
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            const auto approx = search(q, k, ef);
            const auto truth = exact.search(q, k);
            sum += tinyann::recall_at_k(approx, truth);
        }
        return sum / static_cast<double>(queries.size());
    }

    /// Mean recall@k of filtered HNSW search vs filtered exact search.
    template <typename Pred>
    auto recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                        std::size_t k, Pred predicate, std::size_t ef = 0) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value, double> {
        if (ef == 0) {
            ef = params_.ef_search;
        }
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            const auto approx = search(q, k, ef, predicate);
            const auto truth = exact.search(q, k, predicate);
            sum += tinyann::recall_at_k(approx, truth);
        }
        return sum / static_cast<double>(queries.size());
    }

    /// Save compact binary index including the full HNSW graph to path.
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("tinyann::HnswIndex::save: cannot open " + path);
        }
        detail::write_header(out, detail::kKindHnsw);
        detail::write_ids_and_vectors(out, dimension_, metric_, ids_, data_);

        detail::write_pod(out, static_cast<std::uint64_t>(params_.M));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.ef_construction));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.ef_search));
        detail::write_pod(out, params_.seed);
        detail::write_pod(out, static_cast<std::int32_t>(entry_point_));
        detail::write_pod(out, static_cast<std::int32_t>(max_level_));

        const std::size_t n = ids_.size();
        for (std::size_t i = 0; i < n; ++i) {
            detail::write_pod(out, static_cast<std::int32_t>(levels_[i]));
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto& layers = neighbors_[i];
            detail::write_pod(out, static_cast<std::uint32_t>(layers.size()));
            for (const auto& links : layers) {
                detail::write_pod(out, static_cast<std::uint32_t>(links.size()));
                for (int nb : links) {
                    detail::write_pod(out, static_cast<std::int32_t>(nb));
                }
            }
        }

        // Persist RNG so subsequent add() sequences match if the same operations continue.
        std::string rng_state;
        {
            std::ostringstream oss;
            oss << rng_;
            rng_state = oss.str();
        }
        detail::write_pod(out, static_cast<std::uint64_t>(rng_state.size()));
        if (!rng_state.empty()) {
            detail::write_bytes(out, rng_state.data(), rng_state.size());
        }
    }

    /// Load HNSW index from a file written by HnswIndex::save.
    /// Restored graph yields search results identical to the pre-save index.
    /// Rejects corrupt/hostile graphs (OOB entry point, bad levels, invalid neighbor indices).
    static HnswIndex load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("tinyann::HnswIndex::load: cannot open " + path);
        }
        detail::read_header(in, detail::kKindHnsw);

        std::size_t dimension = 0;
        Metric metric = Metric::Cosine;
        std::vector<std::int64_t> ids;
        std::vector<float> data;
        detail::read_ids_and_vectors(in, dimension, metric, ids, data);

        HnswParams params;
        params.M = detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "HNSW M");
        params.ef_construction =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "HNSW ef_construction");
        params.ef_search =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "HNSW ef_search");
        params.seed = detail::read_pod<std::uint64_t>(in);

        HnswIndex idx(dimension, metric, params);
        idx.ids_ = std::move(ids);
        idx.data_ = std::move(data);
        if (idx.data_.size() !=
            detail::checked_mul_size(idx.ids_.size(), idx.dimension_, "hnsw count * dimension")) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (vector storage size mismatch)");
        }
        idx.rebuild_norms();
        idx.entry_point_ = detail::read_pod<std::int32_t>(in);
        idx.max_level_ = detail::read_pod<std::int32_t>(in);

        const std::size_t n = idx.ids_.size();
        // Practical HNSW levels are small (log_M n). Cap rejects hostile huge layer counts
        // before any adjacency allocation.
        constexpr int kMaxHnswLevel = 64;
        constexpr std::size_t kMaxRngStateBytes = 1u << 20;  // 1 MiB

        idx.levels_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const int level = detail::read_pod<std::int32_t>(in);
            if (level < 0 || level > kMaxHnswLevel) {
                throw std::runtime_error(
                    "tinyann::HnswIndex::load: corrupt file (node level out of range)");
            }
            idx.levels_[i] = level;
        }
        idx.neighbors_.assign(n, {});
        for (std::size_t i = 0; i < n; ++i) {
            const auto n_layers = detail::read_pod<std::uint32_t>(in);
            const std::size_t expected_layers =
                static_cast<std::size_t>(idx.levels_[i]) + 1;
            // Enforce layer count before resize (DoS + consistency with levels_[i]).
            if (static_cast<std::size_t>(n_layers) != expected_layers) {
                throw std::runtime_error(
                    "tinyann::HnswIndex::load: corrupt file (neighbors layer count != level+1)");
            }
            idx.neighbors_[i].resize(n_layers);
            for (std::uint32_t lc = 0; lc < n_layers; ++lc) {
                const auto n_links = detail::read_pod<std::uint32_t>(in);
                // A node cannot need more than n neighbor slots (hostile huge lists → DoS).
                if (static_cast<std::size_t>(n_links) > n) {
                    throw std::runtime_error(
                        "tinyann::HnswIndex::load: corrupt file (adjacency list longer than index)");
                }
                idx.neighbors_[i][lc].resize(n_links);
                for (std::uint32_t j = 0; j < n_links; ++j) {
                    idx.neighbors_[i][lc][j] = detail::read_pod<std::int32_t>(in);
                }
            }
        }

        const std::uint64_t rng_len_u64 = detail::read_pod<std::uint64_t>(in);
        if (rng_len_u64 > static_cast<std::uint64_t>(kMaxRngStateBytes)) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (rng state length too large)");
        }
        const auto rng_len = static_cast<std::size_t>(rng_len_u64);
        if (rng_len > 0) {
            std::string rng_state(rng_len, '\0');
            detail::read_bytes(in, &rng_state[0], rng_len);
            std::istringstream iss(rng_state);
            iss >> idx.rng_;
            if (!iss) {
                // Fall back to seed if stream restore fails (should not happen for our writes).
                idx.rng_.seed(params.seed);
            }
        }

        idx.validate_loaded_graph();
        return idx;
    }

private:
    static void validate_params(std::size_t dimension, const HnswParams& params) {
        if (dimension == 0) {
            throw std::invalid_argument("tinyann::HnswIndex: dimension must be > 0");
        }
        if (params.M == 0) {
            throw std::invalid_argument("tinyann::HnswIndex: M must be > 0");
        }
        if (params.ef_construction == 0 || params.ef_search == 0) {
            throw std::invalid_argument("tinyann::HnswIndex: ef_construction/ef_search must be > 0");
        }
    }

    /// Ensure entry point, levels, and adjacency lists are safe to traverse.
    /// Called after binary load; throws runtime_error on corrupt/hostile payloads.
    void validate_loaded_graph() const {
        const std::size_t n = ids_.size();
        if (data_.size() != n * dimension_) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (vector storage size mismatch)");
        }
        if (levels_.size() != n || neighbors_.size() != n || norms_.size() != n) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (per-node arrays size mismatch)");
        }

        if (n == 0) {
            if (entry_point_ != -1 || max_level_ != -1) {
                throw std::runtime_error(
                    "tinyann::HnswIndex::load: corrupt file (empty graph must have "
                    "entry_point=-1 and max_level=-1)");
            }
            return;
        }

        if (entry_point_ < 0 || static_cast<std::size_t>(entry_point_) >= n) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (entry_point out of range)");
        }

        int computed_max = levels_[0];
        for (std::size_t i = 0; i < n; ++i) {
            const int level = levels_[i];
            if (level < 0) {
                throw std::runtime_error(
                    "tinyann::HnswIndex::load: corrupt file (negative node level)");
            }
            if (level > computed_max) {
                computed_max = level;
            }
            // Layer list length must match declared level (layers 0..level inclusive).
            if (neighbors_[i].size() != static_cast<std::size_t>(level) + 1) {
                throw std::runtime_error(
                    "tinyann::HnswIndex::load: corrupt file (neighbors layer count != level+1)");
            }
        }

        if (max_level_ != computed_max) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (max_level inconsistent with node levels)");
        }
        if (levels_[static_cast<std::size_t>(entry_point_)] != max_level_) {
            throw std::runtime_error(
                "tinyann::HnswIndex::load: corrupt file (entry_point not at max_level)");
        }

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t lc = 0; lc < neighbors_[i].size(); ++lc) {
                for (int nb : neighbors_[i][lc]) {
                    if (nb < 0 || static_cast<std::size_t>(nb) >= n) {
                        throw std::runtime_error(
                            "tinyann::HnswIndex::load: corrupt file (neighbor index out of range)");
                    }
                    if (static_cast<std::size_t>(nb) == i) {
                        throw std::runtime_error(
                            "tinyann::HnswIndex::load: corrupt file (self-loop in graph)");
                    }
                    // Neighbor must exist on this layer (has layers 0..lc at least).
                    if (neighbors_[static_cast<std::size_t>(nb)].size() <= lc) {
                        throw std::runtime_error(
                            "tinyann::HnswIndex::load: corrupt file (neighbor missing layer)");
                    }
                }
            }
        }
    }

    /// Hard-delete node at index `idx`, keep graph consistent and searchable.
    void remove_node_at(int idx) {
        const int n = static_cast<int>(ids_.size());
        if (idx < 0 || idx >= n) {
            return;
        }
        const int last = n - 1;
        const bool removed_entry = (entry_point_ == idx);

        // Drop all edges pointing at idx (and clear idx's own lists).
        for (int i = 0; i < n; ++i) {
            for (auto& links : neighbors_[static_cast<std::size_t>(i)]) {
                links.erase(std::remove(links.begin(), links.end(), idx), links.end());
            }
        }

        if (idx != last) {
            // Swap-remove: move last node into the freed slot.
            ids_[static_cast<std::size_t>(idx)] = ids_[static_cast<std::size_t>(last)];
            std::copy(data_.begin() + static_cast<std::ptrdiff_t>(last * static_cast<int>(dimension_)),
                      data_.begin() + static_cast<std::ptrdiff_t>((last + 1) * static_cast<int>(dimension_)),
                      data_.begin() + static_cast<std::ptrdiff_t>(idx * static_cast<int>(dimension_)));
            norms_[static_cast<std::size_t>(idx)] = norms_[static_cast<std::size_t>(last)];
            levels_[static_cast<std::size_t>(idx)] = levels_[static_cast<std::size_t>(last)];
            neighbors_[static_cast<std::size_t>(idx)] =
                std::move(neighbors_[static_cast<std::size_t>(last)]);

            // Remap any neighbor index that still points at `last` -> `idx`.
            // Only nodes 0..last-1 remain meaningful; neighbors_[idx] is the moved last.
            for (int i = 0; i < last; ++i) {
                for (auto& links : neighbors_[static_cast<std::size_t>(i)]) {
                    for (int& nb : links) {
                        if (nb == last) {
                            nb = idx;
                        }
                    }
                }
            }
            if (entry_point_ == last) {
                entry_point_ = idx;
            }
        }

        ids_.pop_back();
        data_.resize(ids_.size() * dimension_);
        norms_.pop_back();
        levels_.pop_back();
        neighbors_.pop_back();

        if (ids_.empty()) {
            entry_point_ = -1;
            max_level_ = -1;
            return;
        }

        if (removed_entry) {
            reassign_entry_point();
        } else {
            // Entry may no longer sit at the global max level after removals.
            recompute_max_level();
            if (entry_point_ < 0 || entry_point_ >= static_cast<int>(ids_.size()) ||
                levels_[static_cast<std::size_t>(entry_point_)] < max_level_) {
                reassign_entry_point();
            }
        }
    }

    void rebuild_norms() {
        norms_.resize(ids_.size());
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            norms_[i] = l2_norm(data_.data() + i * dimension_, dimension_);
        }
    }

    /// Choose entry point as a remaining node with maximum level (min index on ties).
    void reassign_entry_point() {
        entry_point_ = 0;
        max_level_ = levels_[0];
        for (std::size_t i = 1; i < levels_.size(); ++i) {
            if (levels_[i] > max_level_) {
                max_level_ = levels_[i];
                entry_point_ = static_cast<int>(i);
            }
        }
    }

    void recompute_max_level() {
        max_level_ = levels_.empty() ? -1 : levels_[0];
        for (std::size_t i = 1; i < levels_.size(); ++i) {
            if (levels_[i] > max_level_) {
                max_level_ = levels_[i];
            }
        }
    }

    using DistNode = std::pair<float, int>;

    /// Order-preserving graph distance using SIMD kernels + norm caches.
    float distance_nodes(int a, int b) const {
        const float* pa = data_.data() + static_cast<std::size_t>(a) * dimension_;
        const float* pb = data_.data() + static_cast<std::size_t>(b) * dimension_;
        switch (metric_) {
            case Metric::Cosine: {
                const float na = norms_[static_cast<std::size_t>(a)];
                const float nb = norms_[static_cast<std::size_t>(b)];
                if (na == 0.f || nb == 0.f) {
                    return 1.f;  // cosine → 0 ⇒ ann distance 1
                }
                return 1.f - inner_product(pa, pb, dimension_) / (na * nb);
            }
            case Metric::Euclidean:
                return simd::squared_l2(pa, pb, dimension_);
            case Metric::InnerProduct:
                return -inner_product(pa, pb, dimension_);
        }
        return 0.f;
    }

    /// Per-search working memory (stack-allocated / local to each search call).
    /// Must not be shared across threads — this is what makes concurrent `search` safe.
    struct SearchScratch {
        float query_norm = 0.f;
        std::vector<std::uint32_t> visit_mark;
        std::uint32_t visit_tick = 0;

        void begin_visit(std::size_t n) {
            if (visit_mark.size() < n) {
                visit_mark.assign(n, 0);
                visit_tick = 1;
            }
            ++visit_tick;
            if (visit_tick == 0) {
                std::fill(visit_mark.begin(), visit_mark.end(), 0);
                visit_tick = 1;
            }
        }

        bool visited(int node) const {
            return visit_mark[static_cast<std::size_t>(node)] == visit_tick;
        }

        void mark_visited(int node) {
            visit_mark[static_cast<std::size_t>(node)] = visit_tick;
        }
    };

    float distance_query(const float* q, int node, float query_norm) const {
        const float* p = data_.data() + static_cast<std::size_t>(node) * dimension_;
        switch (metric_) {
            case Metric::Cosine: {
                const float nb = norms_[static_cast<std::size_t>(node)];
                if (query_norm == 0.f || nb == 0.f) {
                    return 1.f;
                }
                return 1.f - inner_product(q, p, dimension_) / (query_norm * nb);
            }
            case Metric::Euclidean:
                return simd::squared_l2(q, p, dimension_);
            case Metric::InnerProduct:
                return -inner_product(q, p, dimension_);
        }
        return 0.f;
    }

    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double u = std::max(dist(rng_), 1e-12);
        return static_cast<int>(-std::log(u) * level_mult_);
    }

    int greedy_update(int target, int enter, int layer) const {
        int curr = enter;
        float d = distance_nodes(target, curr);
        bool changed = true;
        while (changed) {
            changed = false;
            const auto& links =
                neighbors_[static_cast<std::size_t>(curr)][static_cast<std::size_t>(layer)];
            for (int nb : links) {
                const float dn = distance_nodes(target, nb);
                if (dn < d) {
                    d = dn;
                    curr = nb;
                    changed = true;
                }
            }
        }
        return curr;
    }

    int greedy_update_query(const float* q, int enter, int layer, SearchScratch& scratch) const {
        int curr = enter;
        float d = distance_query(q, curr, scratch.query_norm);
        bool changed = true;
        while (changed) {
            changed = false;
            const auto& links =
                neighbors_[static_cast<std::size_t>(curr)][static_cast<std::size_t>(layer)];
            for (int nb : links) {
                const float dn = distance_query(q, nb, scratch.query_norm);
                if (dn < d) {
                    d = dn;
                    curr = nb;
                    changed = true;
                }
            }
        }
        return curr;
    }

    std::vector<DistNode> search_layer(int query_node, bool /*query_is_node*/, int enter,
                                       std::size_t ef, int layer) const {
        const float* q = data_.data() + static_cast<std::size_t>(query_node) * dimension_;
        SearchScratch scratch;
        if (metric_ == Metric::Cosine) {
            scratch.query_norm = norms_[static_cast<std::size_t>(query_node)];
        }
        return search_layer_query_filtered(q, enter, ef, layer, scratch,
                                           [](std::int64_t) { return true; });
    }

    /// Layer search with id filter applied to the *eligible result* set only.
    /// Graph edges are still followed through ineligible nodes; those nodes never
    /// enter the result heap — unlike post-filtering an unfiltered top-ef list.
    ///
    /// Expansion rule (tradeoff): a neighbor is expanded only if the eligible heap
    /// has fewer than `ef` hits, or its distance is better than the worst eligible.
    /// Farther ineligible bridges are therefore skipped once the eligible set is full,
    /// which can reduce recall under very selective predicates.
    ///
    /// `scratch` is caller-owned per-search state (not shared across threads).
    template <typename Pred>
    std::vector<DistNode> search_layer_query_filtered(const float* q, int enter, std::size_t ef,
                                                      int layer, SearchScratch& scratch,
                                                      Pred predicate) const {
        scratch.begin_visit(ids_.size());

        // Exploration frontier (all nodes, for connectivity).
        std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates;
        // Eligible dynamic list only (max-heap, size <= ef).
        std::priority_queue<DistNode> eligible;

        const float d0 = distance_query(q, enter, scratch.query_norm);
        candidates.emplace(d0, enter);
        scratch.mark_visited(enter);
        if (predicate(ids_[static_cast<std::size_t>(enter)])) {
            eligible.emplace(d0, enter);
        }

        while (!candidates.empty()) {
            const DistNode c = candidates.top();
            candidates.pop();

            // Stop only when we already have `ef` eligible hits and nothing closer remains.
            // If eligible is still short, keep exploring (important for selective filters).
            if (eligible.size() >= ef && c.first > eligible.top().first) {
                break;
            }

            const auto& links =
                neighbors_[static_cast<std::size_t>(c.second)][static_cast<std::size_t>(layer)];
            for (int nb : links) {
                if (scratch.visited(nb)) {
                    continue;
                }
                scratch.mark_visited(nb);
                const float d = distance_query(q, nb, scratch.query_norm);
                // Expand while eligible list is not full, or this node is closer than
                // the worst eligible (may improve results or path through ineligible).
                if (eligible.size() < ef || d < eligible.top().first) {
                    candidates.emplace(d, nb);
                    if (predicate(ids_[static_cast<std::size_t>(nb)])) {
                        eligible.emplace(d, nb);
                        if (eligible.size() > ef) {
                            eligible.pop();
                        }
                    }
                }
            }
        }

        std::vector<DistNode> out;
        out.reserve(eligible.size());
        while (!eligible.empty()) {
            out.push_back(eligible.top());
            eligible.pop();
        }
        std::sort(out.begin(), out.end(),
                  [](const DistNode& a, const DistNode& b) { return a.first < b.first; });
        return out;
    }

    std::vector<int> select_neighbors(const std::vector<DistNode>& candidates,
                                      std::size_t M) const {
        std::vector<int> selected;
        selected.reserve(std::min(M, candidates.size()));
        for (std::size_t i = 0; i < candidates.size() && selected.size() < M; ++i) {
            selected.push_back(candidates[i].second);
        }
        return selected;
    }

    std::size_t dimension_;
    Metric metric_;
    HnswParams params_;
    // Level sampling in add(); streamed in const save(). Not used by search.
    // Concurrent search is safe while the graph is immutable; concurrent add is not.
    mutable std::mt19937_64 rng_;
    double level_mult_;

    std::vector<std::int64_t> ids_;
    std::vector<float> data_;
    std::vector<float> norms_;  // L2 norms per node (cosine navigation cache)
    std::vector<int> levels_;
    std::vector<std::vector<std::vector<int>>> neighbors_;
    int entry_point_ = -1;
    int max_level_ = -1;
};

// ---------------------------------------------------------------------------
// Approximate index: IVF (Inverted File) — FAISS-style
//
// 1) train(): k-means with `nlist` centroids (metric-aware assignment)
// 2) add(): assign each vector to the nearest centroid's inverted list
// 3) search(): probe `nprobe` closest centroids, brute-force within those lists
// ---------------------------------------------------------------------------

/// Parameters for IVF training and search.
struct IvfParams {
    /// Number of coarse centroids / inverted lists.
    std::size_t nlist = 100;
    /// Number of lists to probe at query time (1 .. nlist).
    std::size_t nprobe = 10;
    /// k-means iterations during train().
    std::size_t kmeans_iters = 25;
    /// RNG seed for centroid initialization.
    std::uint64_t seed = 42;
};

/// In-memory IVF index (k-means coarse quantizer + exact search in probed lists).
///
/// Vectors must be finite. `add` has basic exception safety only. Node indices
/// are `int`; `add` rejects when `size() >= INT_MAX`.
class IvfIndex {
public:
    explicit IvfIndex(std::size_t dimension, Metric metric = Metric::Cosine, IvfParams params = {})
        : dimension_(dimension), metric_(metric), params_(params) {
        if (dimension_ == 0) {
            throw std::invalid_argument("tinyann::IvfIndex: dimension must be > 0");
        }
        if (params_.nlist == 0) {
            throw std::invalid_argument("tinyann::IvfIndex: nlist must be > 0");
        }
        if (params_.nprobe == 0) {
            throw std::invalid_argument("tinyann::IvfIndex: nprobe must be > 0");
        }
        if (params_.kmeans_iters == 0) {
            throw std::invalid_argument("tinyann::IvfIndex: kmeans_iters must be > 0");
        }
    }

    std::size_t dimension() const noexcept { return dimension_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return ids_.size(); }
    bool empty() const noexcept { return ids_.empty(); }
    bool trained() const noexcept { return trained_; }
    const IvfParams& params() const noexcept { return params_; }

    void set_nprobe(std::size_t nprobe) {
        if (nprobe == 0) {
            throw std::invalid_argument("nprobe must be > 0");
        }
        params_.nprobe = nprobe;
    }

    /// Run k-means on training vectors to learn `nlist` centroids.
    /// Clears any previously indexed vectors. Requires non-empty training data.
    void train(const std::vector<std::vector<float>>& training) {
        if (training.empty()) {
            throw std::invalid_argument("tinyann::IvfIndex::train: empty training set");
        }
        for (const auto& v : training) {
            if (v.size() != dimension_) {
                throw std::invalid_argument("tinyann::IvfIndex::train: dimension mismatch");
            }
            require_finite(v, "tinyann::IvfIndex::train");
        }

        // Reset index contents; keep params.
        ids_.clear();
        data_.clear();
        list_of_.clear();
        lists_.assign(params_.nlist, {});
        centroids_.assign(params_.nlist * dimension_, 0.f);

        const std::size_t nt = training.size();
        const std::size_t nlist = std::min(params_.nlist, nt);
        // If training set smaller than nlist, shrink effective lists.
        if (nlist < params_.nlist) {
            params_.nlist = nlist;
            lists_.assign(params_.nlist, {});
            centroids_.assign(params_.nlist * dimension_, 0.f);
        }
        if (params_.nprobe > params_.nlist) {
            params_.nprobe = params_.nlist;
        }

        std::mt19937_64 rng(params_.seed);
        // Init centroids from random distinct training rows (shuffle indices).
        std::vector<std::size_t> order(nt);
        for (std::size_t i = 0; i < nt; ++i) {
            order[i] = i;
        }
        for (std::size_t i = nt; i > 1; --i) {
            std::uniform_int_distribution<std::size_t> dist(0, i - 1);
            std::swap(order[i - 1], order[dist(rng)]);
        }
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            std::copy(training[order[c]].begin(), training[order[c]].end(),
                      centroids_.begin() + static_cast<std::ptrdiff_t>(c * dimension_));
            if (metric_ == Metric::Cosine) {
                normalize_centroid(c);
            }
        }

        std::vector<std::size_t> assign(nt, 0);
        std::vector<std::size_t> counts(params_.nlist, 0);
        std::vector<float> new_cent(params_.nlist * dimension_, 0.f);

        for (std::size_t iter = 0; iter < params_.kmeans_iters; ++iter) {
            // Assign
            for (std::size_t i = 0; i < nt; ++i) {
                assign[i] = nearest_centroid(training[i].data());
            }
            // Accumulate means
            std::fill(new_cent.begin(), new_cent.end(), 0.f);
            std::fill(counts.begin(), counts.end(), 0);
            for (std::size_t i = 0; i < nt; ++i) {
                const std::size_t c = assign[i];
                ++counts[c];
                float* dst = new_cent.data() + c * dimension_;
                const float* src = training[i].data();
                for (std::size_t d = 0; d < dimension_; ++d) {
                    dst[d] += src[d];
                }
            }
            // Update / re-seed empty clusters
            for (std::size_t c = 0; c < params_.nlist; ++c) {
                float* dst = centroids_.data() + c * dimension_;
                if (counts[c] == 0) {
                    std::uniform_int_distribution<std::size_t> dist(0, nt - 1);
                    const auto& v = training[dist(rng)];
                    std::copy(v.begin(), v.end(), dst);
                } else {
                    const float inv = 1.f / static_cast<float>(counts[c]);
                    const float* src = new_cent.data() + c * dimension_;
                    for (std::size_t d = 0; d < dimension_; ++d) {
                        dst[d] = src[d] * inv;
                    }
                }
                if (metric_ == Metric::Cosine) {
                    normalize_centroid(c);
                }
            }
        }

        trained_ = true;
    }

    void add(std::int64_t id, const std::vector<float>& vector) {
        if (!trained_) {
            throw std::runtime_error("tinyann::IvfIndex::add: call train() first");
        }
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IvfIndex::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::IvfIndex::add");
        if (ids_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "tinyann::IvfIndex::add: maximum number of nodes (INT_MAX) reached");
        }
        const int node = static_cast<int>(ids_.size());
        const std::size_t list = nearest_centroid(vector.data());
        ids_.push_back(id);
        data_.insert(data_.end(), vector.begin(), vector.end());
        list_of_.push_back(list);
        lists_[list].push_back(node);
    }

    bool remove(std::int64_t id) {
        bool removed = false;
        for (int i = static_cast<int>(ids_.size()) - 1; i >= 0; --i) {
            if (ids_[static_cast<std::size_t>(i)] == id) {
                remove_node_at(i);
                removed = true;
            }
        }
        return removed;
    }

    bool update(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IvfIndex::update: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::IvfIndex::update");
        bool updated = false;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (ids_[i] != id) {
                continue;
            }
            // Remove from old list
            const std::size_t old_list = list_of_[i];
            auto& lst = lists_[old_list];
            lst.erase(std::remove(lst.begin(), lst.end(), static_cast<int>(i)), lst.end());
            // Replace vector
            std::copy(vector.begin(), vector.end(),
                      data_.begin() + static_cast<std::ptrdiff_t>(i * dimension_));
            // Reassign list
            const std::size_t new_list = nearest_centroid(vector.data());
            list_of_[i] = new_list;
            lists_[new_list].push_back(static_cast<int>(i));
            updated = true;
        }
        return updated;
    }

    bool contains(std::int64_t id) const {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        return search(query, k, [](std::int64_t) { return true; });
    }

    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        if (query.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IvfIndex::search: query dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(query.size()) + ")");
        }
        require_finite(query, "tinyann::IvfIndex::search");
        if (!trained_ || k == 0 || empty()) {
            return {};
        }

        const std::size_t nprobe = std::min(params_.nprobe, params_.nlist);
        auto lists = probe_lists(query.data(), nprobe);

        std::vector<SearchResult> eligible;
        eligible.reserve(k * 2);
        for (std::size_t li : lists) {
            for (int node : lists_[li]) {
                const std::int64_t id = ids_[static_cast<std::size_t>(node)];
                if (!predicate(id)) {
                    continue;
                }
                const float* vec = data_.data() + static_cast<std::size_t>(node) * dimension_;
                eligible.push_back(
                    SearchResult{id, metric_score(metric_, query.data(), vec, dimension_)});
            }
        }
        if (eligible.empty()) {
            return {};
        }

        const std::size_t take = std::min(k, eligible.size());
        const bool hib = higher_is_better(metric_);
        auto better = [hib](const SearchResult& a, const SearchResult& b) {
            if (a.score != b.score) {
                return hib ? (a.score > b.score) : (a.score < b.score);
            }
            return a.id < b.id;
        };
        if (take < eligible.size()) {
            std::partial_sort(eligible.begin(),
                              eligible.begin() + static_cast<std::ptrdiff_t>(take), eligible.end(),
                              better);
            eligible.resize(take);
        } else {
            std::sort(eligible.begin(), eligible.end(), better);
        }
        return eligible;
    }

    double recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                          std::size_t k) const {
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            sum += tinyann::recall_at_k(search(q, k), exact.search(q, k));
        }
        return sum / static_cast<double>(queries.size());
    }

    template <typename Pred>
    auto recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                        std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value, double> {
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            sum += tinyann::recall_at_k(search(q, k, predicate), exact.search(q, k, predicate));
        }
        return sum / static_cast<double>(queries.size());
    }

    void save(const std::string& path) const {
        if (!trained_) {
            throw std::runtime_error("tinyann::IvfIndex::save: not trained");
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("tinyann::IvfIndex::save: cannot open " + path);
        }
        detail::write_header(out, detail::kKindIvf);
        detail::write_ids_and_vectors(out, dimension_, metric_, ids_, data_);
        detail::write_pod(out, static_cast<std::uint64_t>(params_.nlist));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.nprobe));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.kmeans_iters));
        detail::write_pod(out, params_.seed);
        detail::write_bytes(out, centroids_.data(), centroids_.size() * sizeof(float));
        for (std::size_t i = 0; i < list_of_.size(); ++i) {
            detail::write_pod(out, static_cast<std::uint32_t>(list_of_[i]));
        }
        // lists_ can be rebuilt from list_of_; still write for clarity / validation
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            detail::write_pod(out, static_cast<std::uint32_t>(lists_[c].size()));
            for (int node : lists_[c]) {
                detail::write_pod(out, static_cast<std::int32_t>(node));
            }
        }
    }

    /// Load IVF index from a file written by IvfIndex::save.
    /// Rejects corrupt/hostile payloads (bad list ids, OOB node indices, list/list_of mismatch).
    static IvfIndex load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("tinyann::IvfIndex::load: cannot open " + path);
        }
        detail::read_header(in, detail::kKindIvf);
        std::size_t dimension = 0;
        Metric metric = Metric::Cosine;
        std::vector<std::int64_t> ids;
        std::vector<float> data;
        detail::read_ids_and_vectors(in, dimension, metric, ids, data);

        IvfParams params;
        params.nlist = detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "IVF nlist");
        params.nprobe = detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "IVF nprobe");
        params.kmeans_iters =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "IVF kmeans_iters");
        params.seed = detail::read_pod<std::uint64_t>(in);

        // Constructor validates nlist/nprobe/kmeans_iters > 0.
        IvfIndex idx(dimension, metric, params);
        idx.ids_ = std::move(ids);
        idx.data_ = std::move(data);
        if (idx.data_.size() !=
            detail::checked_mul_size(idx.ids_.size(), idx.dimension_, "ivf count * dimension")) {
            throw std::runtime_error(
                "tinyann::IvfIndex::load: corrupt file (vector storage size mismatch)");
        }
        const std::size_t n_cent =
            detail::checked_mul_size(params.nlist, dimension, "nlist * dimension (centroids)");
        const std::size_t cent_bytes =
            detail::checked_mul_size(n_cent, sizeof(float), "centroids bytes");
        idx.centroids_.resize(n_cent);
        detail::read_bytes(in, idx.centroids_.data(), cent_bytes);
        idx.list_of_.resize(idx.ids_.size());
        for (std::size_t i = 0; i < idx.list_of_.size(); ++i) {
            idx.list_of_[i] = detail::read_pod<std::uint32_t>(in);
        }
        idx.lists_.assign(params.nlist, {});
        for (std::size_t c = 0; c < params.nlist; ++c) {
            const auto sz = static_cast<std::size_t>(detail::read_pod<std::uint32_t>(in));
            // Bound list length: a list cannot hold more than n nodes (even with dups rejected later).
            if (sz > idx.ids_.size()) {
                throw std::runtime_error(
                    "tinyann::IvfIndex::load: corrupt file (inverted list longer than index)");
            }
            idx.lists_[c].resize(sz);
            for (std::size_t j = 0; j < sz; ++j) {
                idx.lists_[c][j] = detail::read_pod<std::int32_t>(in);
            }
        }
        idx.validate_loaded_lists();
        // Canonical inverted lists from validated list_of_ (file lists only for cross-check).
        idx.rebuild_lists_from_list_of();
        if (idx.params_.nprobe > idx.params_.nlist) {
            idx.params_.nprobe = idx.params_.nlist;
        }
        idx.trained_ = true;
        return idx;
    }

private:
    /// Ensure list_of_ / lists_ / centroids are safe to search and mutate.
    /// Called after binary load; throws runtime_error on corrupt/hostile payloads.
    void validate_loaded_lists() const {
        const std::size_t n = ids_.size();
        if (data_.size() != n * dimension_) {
            throw std::runtime_error(
                "tinyann::IvfIndex::load: corrupt file (vector storage size mismatch)");
        }
        if (list_of_.size() != n) {
            throw std::runtime_error(
                "tinyann::IvfIndex::load: corrupt file (list_of size mismatch)");
        }
        if (params_.nlist == 0 || lists_.size() != params_.nlist) {
            throw std::runtime_error(
                "tinyann::IvfIndex::load: corrupt file (lists / nlist mismatch)");
        }
        if (centroids_.size() != params_.nlist * dimension_) {
            throw std::runtime_error(
                "tinyann::IvfIndex::load: corrupt file (centroids size mismatch)");
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (list_of_[i] >= params_.nlist) {
                throw std::runtime_error(
                    "tinyann::IvfIndex::load: corrupt file (list_of entry out of range)");
            }
        }

        // Cross-check serialized inverted lists against list_of_: every node appears exactly
        // once, only in the list claimed by list_of_[node].
        std::vector<std::uint8_t> seen(n, 0);
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            for (int node : lists_[c]) {
                if (node < 0 || static_cast<std::size_t>(node) >= n) {
                    throw std::runtime_error(
                        "tinyann::IvfIndex::load: corrupt file (list node index out of range)");
                }
                const std::size_t u = static_cast<std::size_t>(node);
                if (list_of_[u] != c) {
                    throw std::runtime_error(
                        "tinyann::IvfIndex::load: corrupt file (lists disagree with list_of)");
                }
                if (seen[u] != 0) {
                    throw std::runtime_error(
                        "tinyann::IvfIndex::load: corrupt file (duplicate node in lists)");
                }
                seen[u] = 1;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (seen[i] == 0) {
                throw std::runtime_error(
                    "tinyann::IvfIndex::load: corrupt file (node missing from lists)");
            }
        }
    }

    /// Rebuild inverted lists from list_of_ (source of truth after validation).
    void rebuild_lists_from_list_of() {
        lists_.assign(params_.nlist, {});
        for (std::size_t i = 0; i < list_of_.size(); ++i) {
            lists_[list_of_[i]].push_back(static_cast<int>(i));
        }
    }

    void normalize_centroid(std::size_t c) {
        float* p = centroids_.data() + c * dimension_;
        const float n = l2_norm(p, dimension_);
        if (n > 0.f) {
            const float inv = 1.f / n;
            for (std::size_t d = 0; d < dimension_; ++d) {
                p[d] *= inv;
            }
        }
    }

    /// True if score `a` is a better assignment than `b` for the metric.
    static bool better_score(Metric m, float a, float b) {
        return higher_is_better(m) ? (a > b) : (a < b);
    }

    std::size_t nearest_centroid(const float* v) const {
        std::size_t best = 0;
        float best_s = metric_score(metric_, v, centroids_.data(), dimension_);
        for (std::size_t c = 1; c < params_.nlist; ++c) {
            const float s =
                metric_score(metric_, v, centroids_.data() + c * dimension_, dimension_);
            if (better_score(metric_, s, best_s)) {
                best_s = s;
                best = c;
            }
        }
        return best;
    }

    std::vector<std::size_t> probe_lists(const float* q, std::size_t nprobe) const {
        std::vector<std::pair<float, std::size_t>> scored;
        scored.reserve(params_.nlist);
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            const float s =
                metric_score(metric_, q, centroids_.data() + c * dimension_, dimension_);
            scored.emplace_back(s, c);
        }
        const bool hib = higher_is_better(metric_);
        const std::size_t take = std::min(nprobe, scored.size());
        auto better = [hib](const std::pair<float, std::size_t>& a,
                            const std::pair<float, std::size_t>& b) {
            if (a.first != b.first) {
                return hib ? (a.first > b.first) : (a.first < b.first);
            }
            return a.second < b.second;
        };
        if (take < scored.size()) {
            std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(take),
                              scored.end(), better);
            scored.resize(take);
        } else {
            std::sort(scored.begin(), scored.end(), better);
        }
        std::vector<std::size_t> out;
        out.reserve(scored.size());
        for (const auto& p : scored) {
            out.push_back(p.second);
        }
        return out;
    }

    void remove_node_at(int idx) {
        const int n = static_cast<int>(ids_.size());
        if (idx < 0 || idx >= n) {
            return;
        }
        const int last = n - 1;
        const std::size_t list = list_of_[static_cast<std::size_t>(idx)];
        auto& lst = lists_[list];
        lst.erase(std::remove(lst.begin(), lst.end(), idx), lst.end());

        if (idx != last) {
            // Remove last from its list (will re-insert as idx)
            const std::size_t last_list = list_of_[static_cast<std::size_t>(last)];
            auto& ll = lists_[last_list];
            ll.erase(std::remove(ll.begin(), ll.end(), last), ll.end());

            ids_[static_cast<std::size_t>(idx)] = ids_[static_cast<std::size_t>(last)];
            std::copy(data_.begin() + static_cast<std::ptrdiff_t>(last * static_cast<int>(dimension_)),
                      data_.begin() + static_cast<std::ptrdiff_t>((last + 1) * static_cast<int>(dimension_)),
                      data_.begin() + static_cast<std::ptrdiff_t>(idx * static_cast<int>(dimension_)));
            list_of_[static_cast<std::size_t>(idx)] = list_of_[static_cast<std::size_t>(last)];
            lists_[list_of_[static_cast<std::size_t>(idx)]].push_back(idx);
        }

        ids_.pop_back();
        data_.resize(ids_.size() * dimension_);
        list_of_.pop_back();
    }

    std::size_t dimension_;
    Metric metric_;
    IvfParams params_;
    bool trained_ = false;

    std::vector<float> centroids_;  // nlist * dim
    std::vector<std::vector<int>> lists_;
    std::vector<std::int64_t> ids_;
    std::vector<float> data_;
    std::vector<std::size_t> list_of_;  // node -> list id
};

// ---------------------------------------------------------------------------
// Scalar quantization (SQ, int8) — FAISS-style symmetric per-vector scales
// x_i ≈ scale * code_i,  code_i ∈ [-127, 127]
// ---------------------------------------------------------------------------

namespace sq {

/// Symmetric int8 encode: scale = max|x| / 127 (scale=1 if vector is zero).
inline void quantize(const float* v, std::size_t dim, std::int8_t* codes, float& scale_out) {
    float max_abs = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float a = std::fabs(v[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    if (max_abs == 0.f) {
        scale_out = 1.f;
        std::fill(codes, codes + dim, static_cast<std::int8_t>(0));
        return;
    }
    scale_out = max_abs / 127.f;
    const float inv = 1.f / scale_out;
    for (std::size_t i = 0; i < dim; ++i) {
        const float q = std::round(v[i] * inv);
        const int qi = static_cast<int>(q);
        const int clamped = std::max(-127, std::min(127, qi));
        codes[i] = static_cast<std::int8_t>(clamped);
    }
}

/// Decode: out_i = scale * code_i
inline void dequantize(const std::int8_t* codes, float scale, std::size_t dim, float* out) {
    for (std::size_t i = 0; i < dim; ++i) {
        out[i] = scale * static_cast<float>(codes[i]);
    }
}

/// Convenience: quantize into vectors.
inline void quantize(const std::vector<float>& v, std::vector<std::int8_t>& codes, float& scale) {
    codes.resize(v.size());
    quantize(v.data(), v.size(), codes.data(), scale);
}

inline std::vector<float> dequantize(const std::vector<std::int8_t>& codes, float scale) {
    std::vector<float> out(codes.size());
    dequantize(codes.data(), scale, codes.size(), out.data());
    return out;
}

}  // namespace sq

/// Exact k-NN over **int8 scalar-quantized** vectors (per-vector symmetric scale).
/// Distances use dequantized floats and the same metrics as `Index`.
/// Finite input vectors required on `add`/`update`/`search`. Basic exception safety on `add`.
class IndexSq {
public:
    explicit IndexSq(std::size_t dimension, Metric metric = Metric::Cosine)
        : dimension_(dimension), metric_(metric) {
        if (dimension_ == 0) {
            throw std::invalid_argument("tinyann::IndexSq: dimension must be > 0");
        }
    }

    std::size_t dimension() const noexcept { return dimension_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return ids_.size(); }
    bool empty() const noexcept { return ids_.empty(); }

    /// Quantize and store (int8 codes + scale). Basic exception safety only.
    void add(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IndexSq::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::IndexSq::add");
        float scale = 1.f;
        const std::size_t off = codes_.size();
        codes_.resize(off + dimension_);
        sq::quantize(vector.data(), dimension_, codes_.data() + off, scale);
        scales_.push_back(scale);
        ids_.push_back(id);
    }

    bool remove(std::int64_t id) {
        bool removed = false;
        for (std::size_t i = 0; i < ids_.size();) {
            if (ids_[i] != id) {
                ++i;
                continue;
            }
            const std::size_t last = ids_.size() - 1;
            if (i != last) {
                ids_[i] = ids_[last];
                scales_[i] = scales_[last];
                std::copy(codes_.begin() + static_cast<std::ptrdiff_t>(last * dimension_),
                          codes_.begin() + static_cast<std::ptrdiff_t>((last + 1) * dimension_),
                          codes_.begin() + static_cast<std::ptrdiff_t>(i * dimension_));
            }
            ids_.pop_back();
            scales_.pop_back();
            codes_.resize(ids_.size() * dimension_);
            removed = true;
        }
        return removed;
    }

    bool update(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IndexSq::update: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::IndexSq::update");
        bool updated = false;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (ids_[i] != id) {
                continue;
            }
            float scale = 1.f;
            sq::quantize(vector.data(), dimension_,
                         codes_.data() + i * dimension_, scale);
            scales_[i] = scale;
            updated = true;
        }
        return updated;
    }

    bool contains(std::int64_t id) const {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    /// Dequantize vector `i` into `out` (size == dimension()).
    void reconstruct(std::size_t i, float* out) const {
        if (i >= ids_.size()) {
            throw std::out_of_range("tinyann::IndexSq::reconstruct");
        }
        sq::dequantize(codes_.data() + i * dimension_, scales_[i], dimension_, out);
    }

    std::vector<float> reconstruct(std::size_t i) const {
        std::vector<float> out(dimension_);
        reconstruct(i, out.data());
        return out;
    }

    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        return search(query, k, [](std::int64_t) { return true; });
    }

    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        if (query.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IndexSq::search: query dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(query.size()) + ")");
        }
        require_finite(query, "tinyann::IndexSq::search");
        if (k == 0 || empty()) {
            return {};
        }

        std::vector<float> recon(dimension_);
        std::vector<SearchResult> eligible;
        eligible.reserve(std::min(k, size()));
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (!predicate(ids_[i])) {
                continue;
            }
            sq::dequantize(codes_.data() + i * dimension_, scales_[i], dimension_, recon.data());
            eligible.push_back(SearchResult{
                ids_[i], metric_score(metric_, query.data(), recon.data(), dimension_)});
        }
        if (eligible.empty()) {
            return {};
        }

        const std::size_t take = std::min(k, eligible.size());
        const bool hib = higher_is_better(metric_);
        auto better = [hib](const SearchResult& a, const SearchResult& b) {
            if (a.score != b.score) {
                return hib ? (a.score > b.score) : (a.score < b.score);
            }
            return a.id < b.id;
        };
        if (take < eligible.size()) {
            std::partial_sort(eligible.begin(),
                              eligible.begin() + static_cast<std::ptrdiff_t>(take), eligible.end(),
                              better);
            eligible.resize(take);
        } else {
            std::sort(eligible.begin(), eligible.end(), better);
        }
        return eligible;
    }

    double recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                          std::size_t k) const {
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            sum += tinyann::recall_at_k(search(q, k), exact.search(q, k));
        }
        return sum / static_cast<double>(queries.size());
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("tinyann::IndexSq::save: cannot open " + path);
        }
        detail::write_header(out, detail::kKindSq);
        detail::write_pod(out, static_cast<std::uint32_t>(metric_));
        detail::write_pod(out, static_cast<std::uint64_t>(dimension_));
        detail::write_pod(out, static_cast<std::uint64_t>(ids_.size()));
        if (!ids_.empty()) {
            detail::write_bytes(out, ids_.data(), ids_.size() * sizeof(std::int64_t));
            detail::write_bytes(out, scales_.data(), scales_.size() * sizeof(float));
            detail::write_bytes(out, codes_.data(), codes_.size() * sizeof(std::int8_t));
        }
    }

    static IndexSq load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("tinyann::IndexSq::load: cannot open " + path);
        }
        detail::read_header(in, detail::kKindSq);
        const Metric metric = detail::metric_from_u32(detail::read_pod<std::uint32_t>(in));
        const std::size_t dimension =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "sq dimension");
        const std::size_t count =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "sq vector count");
        if (dimension == 0) {
            throw std::runtime_error("tinyann::IndexSq::load: corrupt (dimension == 0)");
        }
        const std::size_t n_codes =
            detail::checked_mul_size(count, dimension, "sq count * dimension");
        const std::size_t ids_bytes =
            detail::checked_mul_size(count, sizeof(std::int64_t), "sq ids bytes");
        const std::size_t scales_bytes =
            detail::checked_mul_size(count, sizeof(float), "sq scales bytes");
        // codes are int8: byte count == n_codes
        IndexSq idx(dimension, metric);
        idx.ids_.resize(count);
        idx.scales_.resize(count);
        idx.codes_.resize(n_codes);
        if (count > 0) {
            detail::read_bytes(in, idx.ids_.data(), ids_bytes);
            detail::read_bytes(in, idx.scales_.data(), scales_bytes);
            detail::read_bytes(in, idx.codes_.data(), n_codes);
        }
        return idx;
    }

    const std::vector<std::int64_t>& ids() const noexcept { return ids_; }
    const std::vector<float>& scales() const noexcept { return scales_; }
    const std::vector<std::int8_t>& codes() const noexcept { return codes_; }

private:
    std::size_t dimension_;
    Metric metric_;
    std::vector<std::int64_t> ids_;
    std::vector<float> scales_;       // per-vector symmetric scale
    std::vector<std::int8_t> codes_;  // row-major int8
};

// ---------------------------------------------------------------------------
// IVF + Product Quantization (IVFPQ) — FAISS-style compressed inverted index
//
// 1) train(): coarse k-means (nlist) + M independent subspace codebooks (ksub=256)
//    on residuals to the coarse centroid (IVFADC residual PQ)
// 2) add(): assign list, encode residual as M uint8 codes (no full-float storage)
// 3) search(): probe nprobe lists; asymmetric distance computation (ADC) vs codes
//
// Memory: ~M bytes/vector + coarse centroids + M*256*(dim/M) floats for codebooks
// ---------------------------------------------------------------------------

/// Parameters for IVFPQ training and search.
struct IvfPqParams {
    /// Number of coarse centroids / inverted lists.
    std::size_t nlist = 100;
    /// Number of lists to probe at query time (1 .. nlist).
    std::size_t nprobe = 10;
    /// k-means iterations for the coarse quantizer.
    std::size_t kmeans_iters = 25;
    /// Number of PQ subquantizers (must divide dimension).
    std::size_t M = 8;
    /// k-means iterations per PQ subspace codebook.
    std::size_t pq_kmeans_iters = 25;
    /// RNG seed for coarse + PQ training.
    std::uint64_t seed = 42;
};

/// In-memory IVF index with product-quantized residuals (compressed corpus).
///
/// Stores only M-byte codes per vector (plus inverted lists / centroids / codebooks).
/// Suitable for large corpora where full-float IVF or HNSW RAM is prohibitive.
/// Concurrent search is safe while the index is immutable (no concurrent train/add).
class IvfPqIndex {
public:
    /// Fixed 8-bit codes per subspace (FAISS default ksub = 256).
    static constexpr std::size_t kSub = 256;

    explicit IvfPqIndex(std::size_t dimension, Metric metric = Metric::Cosine,
                        IvfPqParams params = {})
        : dimension_(dimension), metric_(metric), params_(params) {
        validate_ctor();
    }

    std::size_t dimension() const noexcept { return dimension_; }
    Metric metric() const noexcept { return metric_; }
    std::size_t size() const noexcept { return ids_.size(); }
    bool empty() const noexcept { return ids_.empty(); }
    bool trained() const noexcept { return trained_; }
    const IvfPqParams& params() const noexcept { return params_; }
    std::size_t code_size() const noexcept { return params_.M; }  // bytes per vector

    void set_nprobe(std::size_t nprobe) {
        if (nprobe == 0) {
            throw std::invalid_argument("nprobe must be > 0");
        }
        params_.nprobe = nprobe;
    }

    /// Train coarse IVF centroids and PQ codebooks on residuals. Clears any vectors.
    void train(const std::vector<std::vector<float>>& training) {
        if (training.empty()) {
            throw std::invalid_argument("tinyann::IvfPqIndex::train: empty training set");
        }
        for (const auto& v : training) {
            if (v.size() != dimension_) {
                throw std::invalid_argument("tinyann::IvfPqIndex::train: dimension mismatch");
            }
            require_finite(v, "tinyann::IvfPqIndex::train");
        }

        clear_data();
        train_coarse(training);
        train_pq_codebooks(training);
        trained_ = true;
    }

    void add(std::int64_t id, const std::vector<float>& vector) {
        if (!trained_) {
            throw std::runtime_error("tinyann::IvfPqIndex::add: call train() first");
        }
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IvfPqIndex::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::IvfPqIndex::add");
        if (ids_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "tinyann::IvfPqIndex::add: maximum number of nodes (INT_MAX) reached");
        }

        const int node = static_cast<int>(ids_.size());
        const std::size_t list = nearest_centroid(vector.data());
        std::vector<std::uint8_t> code(params_.M);
        encode_residual(vector.data(), list, code.data());

        ids_.push_back(id);
        list_of_.push_back(list);
        lists_[list].push_back(node);
        codes_.insert(codes_.end(), code.begin(), code.end());
    }

    bool remove(std::int64_t id) {
        bool removed = false;
        for (int i = static_cast<int>(ids_.size()) - 1; i >= 0; --i) {
            if (ids_[static_cast<std::size_t>(i)] == id) {
                remove_node_at(i);
                removed = true;
            }
        }
        return removed;
    }

    bool update(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IvfPqIndex::update: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        require_finite(vector, "tinyann::IvfPqIndex::update");
        bool updated = false;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            if (ids_[i] != id) {
                continue;
            }
            const std::size_t old_list = list_of_[i];
            auto& lst = lists_[old_list];
            lst.erase(std::remove(lst.begin(), lst.end(), static_cast<int>(i)), lst.end());

            const std::size_t new_list = nearest_centroid(vector.data());
            list_of_[i] = new_list;
            lists_[new_list].push_back(static_cast<int>(i));
            encode_residual(vector.data(), new_list, codes_.data() + i * params_.M);
            updated = true;
        }
        return updated;
    }

    bool contains(std::int64_t id) const {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        return search(query, k, [](std::int64_t) { return true; });
    }

    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        if (query.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::IvfPqIndex::search: query dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(query.size()) + ")");
        }
        require_finite(query, "tinyann::IvfPqIndex::search");
        if (!trained_ || k == 0 || empty()) {
            return {};
        }

        const std::size_t nprobe = std::min(params_.nprobe, params_.nlist);
        auto lists = probe_lists(query.data(), nprobe);

        std::vector<SearchResult> eligible;
        eligible.reserve(k * 4);

        // ADC tables: M * kSub floats (rebuilt per probed list because residual depends on
        // the coarse centroid).
        std::vector<float> table(params_.M * kSub);
        std::vector<float> residual(dimension_);

        for (std::size_t li : lists) {
            build_adc_table(query.data(), li, residual.data(), table.data());
            const float list_base = list_base_score(query.data(), li);
            for (int node : lists_[li]) {
                const std::int64_t id = ids_[static_cast<std::size_t>(node)];
                if (!predicate(id)) {
                    continue;
                }
                const std::uint8_t* code = codes_.data() + static_cast<std::size_t>(node) * params_.M;
                const float score = list_base + adc_from_table(code, table.data());
                eligible.push_back(SearchResult{id, score});
            }
        }
        if (eligible.empty()) {
            return {};
        }

        const std::size_t take = std::min(k, eligible.size());
        const bool hib = higher_is_better(metric_);
        auto better = [hib](const SearchResult& a, const SearchResult& b) {
            if (a.score != b.score) {
                return hib ? (a.score > b.score) : (a.score < b.score);
            }
            return a.id < b.id;
        };
        if (take < eligible.size()) {
            std::partial_sort(eligible.begin(),
                              eligible.begin() + static_cast<std::ptrdiff_t>(take), eligible.end(),
                              better);
            eligible.resize(take);
        } else {
            std::sort(eligible.begin(), eligible.end(), better);
        }
        return eligible;
    }

    double recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                          std::size_t k) const {
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            sum += tinyann::recall_at_k(search(q, k), exact.search(q, k));
        }
        return sum / static_cast<double>(queries.size());
    }

    template <typename Pred>
    auto recall_at_k_vs(const Index& exact, const std::vector<std::vector<float>>& queries,
                        std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value, double> {
        if (queries.empty()) {
            return 1.0;
        }
        double sum = 0.0;
        for (const auto& q : queries) {
            sum += tinyann::recall_at_k(search(q, k, predicate), exact.search(q, k, predicate));
        }
        return sum / static_cast<double>(queries.size());
    }

    void save(const std::string& path) const {
        if (!trained_) {
            throw std::runtime_error("tinyann::IvfPqIndex::save: not trained");
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("tinyann::IvfPqIndex::save: cannot open " + path);
        }
        detail::write_header(out, detail::kKindIvfPq);
        detail::write_pod(out, static_cast<std::uint32_t>(metric_));
        detail::write_pod(out, static_cast<std::uint64_t>(dimension_));
        detail::write_pod(out, static_cast<std::uint64_t>(ids_.size()));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.nlist));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.nprobe));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.kmeans_iters));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.M));
        detail::write_pod(out, static_cast<std::uint64_t>(params_.pq_kmeans_iters));
        detail::write_pod(out, params_.seed);

        if (!ids_.empty()) {
            detail::write_bytes(out, ids_.data(), ids_.size() * sizeof(std::int64_t));
        }
        detail::write_bytes(out, centroids_.data(), centroids_.size() * sizeof(float));
        detail::write_bytes(out, codebooks_.data(), codebooks_.size() * sizeof(float));
        for (std::size_t i = 0; i < list_of_.size(); ++i) {
            detail::write_pod(out, static_cast<std::uint32_t>(list_of_[i]));
        }
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            detail::write_pod(out, static_cast<std::uint32_t>(lists_[c].size()));
            for (int node : lists_[c]) {
                detail::write_pod(out, static_cast<std::int32_t>(node));
            }
        }
        if (!codes_.empty()) {
            detail::write_bytes(out, codes_.data(), codes_.size() * sizeof(std::uint8_t));
        }
    }

    static IvfPqIndex load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("tinyann::IvfPqIndex::load: cannot open " + path);
        }
        detail::read_header(in, detail::kKindIvfPq);
        const Metric metric = detail::metric_from_u32(detail::read_pod<std::uint32_t>(in));
        const std::size_t dimension =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq dimension");
        const std::size_t count =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq count");

        IvfPqParams params;
        params.nlist = detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq nlist");
        params.nprobe = detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq nprobe");
        params.kmeans_iters =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq kmeans_iters");
        params.M = detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq M");
        params.pq_kmeans_iters =
            detail::size_from_u64(detail::read_pod<std::uint64_t>(in), "ivfpq pq_kmeans_iters");
        params.seed = detail::read_pod<std::uint64_t>(in);

        IvfPqIndex idx(dimension, metric, params);
        const std::size_t dsub = dimension / params.M;

        idx.ids_.resize(count);
        if (count > 0) {
            const std::size_t ids_bytes =
                detail::checked_mul_size(count, sizeof(std::int64_t), "ivfpq ids bytes");
            detail::read_bytes(in, idx.ids_.data(), ids_bytes);
        }

        const std::size_t n_cent =
            detail::checked_mul_size(params.nlist, dimension, "ivfpq centroids");
        idx.centroids_.resize(n_cent);
        detail::read_bytes(in, idx.centroids_.data(),
                           detail::checked_mul_size(n_cent, sizeof(float), "ivfpq cent bytes"));

        const std::size_t n_cb = detail::checked_mul_size(
            detail::checked_mul_size(params.M, kSub, "ivfpq M*ksub"), dsub, "ivfpq codebooks");
        idx.codebooks_.resize(n_cb);
        detail::read_bytes(in, idx.codebooks_.data(),
                           detail::checked_mul_size(n_cb, sizeof(float), "ivfpq codebook bytes"));

        idx.list_of_.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            idx.list_of_[i] = detail::read_pod<std::uint32_t>(in);
        }
        idx.lists_.assign(params.nlist, {});
        for (std::size_t c = 0; c < params.nlist; ++c) {
            const auto sz = static_cast<std::size_t>(detail::read_pod<std::uint32_t>(in));
            if (sz > count) {
                throw std::runtime_error(
                    "tinyann::IvfPqIndex::load: corrupt file (list longer than index)");
            }
            idx.lists_[c].resize(sz);
            for (std::size_t j = 0; j < sz; ++j) {
                idx.lists_[c][j] = detail::read_pod<std::int32_t>(in);
            }
        }

        const std::size_t n_codes = detail::checked_mul_size(count, params.M, "ivfpq codes");
        idx.codes_.resize(n_codes);
        if (n_codes > 0) {
            detail::read_bytes(in, idx.codes_.data(), n_codes);
        }

        idx.validate_loaded();
        idx.rebuild_lists_from_list_of();
        if (idx.params_.nprobe > idx.params_.nlist) {
            idx.params_.nprobe = idx.params_.nlist;
        }
        idx.trained_ = true;
        return idx;
    }

    const std::vector<std::int64_t>& ids() const noexcept { return ids_; }
    const std::vector<std::uint8_t>& codes() const noexcept { return codes_; }

private:
    void validate_ctor() {
        if (dimension_ == 0) {
            throw std::invalid_argument("tinyann::IvfPqIndex: dimension must be > 0");
        }
        if (params_.nlist == 0) {
            throw std::invalid_argument("tinyann::IvfPqIndex: nlist must be > 0");
        }
        if (params_.nprobe == 0) {
            throw std::invalid_argument("tinyann::IvfPqIndex: nprobe must be > 0");
        }
        if (params_.kmeans_iters == 0 || params_.pq_kmeans_iters == 0) {
            throw std::invalid_argument("tinyann::IvfPqIndex: kmeans iters must be > 0");
        }
        if (params_.M == 0) {
            throw std::invalid_argument("tinyann::IvfPqIndex: M must be > 0");
        }
        if (dimension_ % params_.M != 0) {
            throw std::invalid_argument(
                "tinyann::IvfPqIndex: dimension must be divisible by M (got dim=" +
                std::to_string(dimension_) + ", M=" + std::to_string(params_.M) + ")");
        }
    }

    void clear_data() {
        ids_.clear();
        codes_.clear();
        list_of_.clear();
        lists_.assign(params_.nlist, {});
        centroids_.clear();
        codebooks_.clear();
        trained_ = false;
    }

    std::size_t dsub() const noexcept { return dimension_ / params_.M; }

    /// Codebook layout: codebooks_[m * kSub * dsub + k * dsub + d]
    float* codebook_entry(std::size_t m, std::size_t k) {
        return codebooks_.data() + (m * kSub + k) * dsub();
    }
    const float* codebook_entry(std::size_t m, std::size_t k) const {
        return codebooks_.data() + (m * kSub + k) * dsub();
    }

    static bool better_score(Metric m, float a, float b) {
        return higher_is_better(m) ? (a > b) : (a < b);
    }

    void normalize_centroid(std::size_t c) {
        float* p = centroids_.data() + c * dimension_;
        const float n = l2_norm(p, dimension_);
        if (n > 0.f) {
            const float inv = 1.f / n;
            for (std::size_t d = 0; d < dimension_; ++d) {
                p[d] *= inv;
            }
        }
    }

    std::size_t nearest_centroid(const float* v) const {
        std::size_t best = 0;
        float best_s = metric_score(metric_, v, centroids_.data(), dimension_);
        for (std::size_t c = 1; c < params_.nlist; ++c) {
            const float s =
                metric_score(metric_, v, centroids_.data() + c * dimension_, dimension_);
            if (better_score(metric_, s, best_s)) {
                best_s = s;
                best = c;
            }
        }
        return best;
    }

    std::vector<std::size_t> probe_lists(const float* q, std::size_t nprobe) const {
        std::vector<std::pair<float, std::size_t>> scored;
        scored.reserve(params_.nlist);
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            const float s =
                metric_score(metric_, q, centroids_.data() + c * dimension_, dimension_);
            scored.emplace_back(s, c);
        }
        const bool hib = higher_is_better(metric_);
        const std::size_t take = std::min(nprobe, scored.size());
        auto better = [hib](const std::pair<float, std::size_t>& a,
                            const std::pair<float, std::size_t>& b) {
            if (a.first != b.first) {
                return hib ? (a.first > b.first) : (a.first < b.first);
            }
            return a.second < b.second;
        };
        if (take < scored.size()) {
            std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(take),
                              scored.end(), better);
            scored.resize(take);
        } else {
            std::sort(scored.begin(), scored.end(), better);
        }
        std::vector<std::size_t> out;
        out.reserve(scored.size());
        for (const auto& p : scored) {
            out.push_back(p.second);
        }
        return out;
    }

    void train_coarse(const std::vector<std::vector<float>>& training) {
        const std::size_t nt = training.size();
        const std::size_t nlist = std::min(params_.nlist, nt);
        if (nlist < params_.nlist) {
            params_.nlist = nlist;
        }
        if (params_.nprobe > params_.nlist) {
            params_.nprobe = params_.nlist;
        }
        lists_.assign(params_.nlist, {});
        centroids_.assign(params_.nlist * dimension_, 0.f);

        std::mt19937_64 rng(params_.seed);
        std::vector<std::size_t> order(nt);
        for (std::size_t i = 0; i < nt; ++i) {
            order[i] = i;
        }
        for (std::size_t i = nt; i > 1; --i) {
            std::uniform_int_distribution<std::size_t> dist(0, i - 1);
            std::swap(order[i - 1], order[dist(rng)]);
        }
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            std::copy(training[order[c]].begin(), training[order[c]].end(),
                      centroids_.begin() + static_cast<std::ptrdiff_t>(c * dimension_));
            if (metric_ == Metric::Cosine) {
                normalize_centroid(c);
            }
        }

        std::vector<std::size_t> assign(nt, 0);
        std::vector<std::size_t> counts(params_.nlist, 0);
        std::vector<float> new_cent(params_.nlist * dimension_, 0.f);

        for (std::size_t iter = 0; iter < params_.kmeans_iters; ++iter) {
            for (std::size_t i = 0; i < nt; ++i) {
                assign[i] = nearest_centroid(training[i].data());
            }
            std::fill(new_cent.begin(), new_cent.end(), 0.f);
            std::fill(counts.begin(), counts.end(), 0);
            for (std::size_t i = 0; i < nt; ++i) {
                const std::size_t c = assign[i];
                ++counts[c];
                float* dst = new_cent.data() + c * dimension_;
                const float* src = training[i].data();
                for (std::size_t d = 0; d < dimension_; ++d) {
                    dst[d] += src[d];
                }
            }
            for (std::size_t c = 0; c < params_.nlist; ++c) {
                float* dst = centroids_.data() + c * dimension_;
                if (counts[c] == 0) {
                    std::uniform_int_distribution<std::size_t> dist(0, nt - 1);
                    const auto& v = training[dist(rng)];
                    std::copy(v.begin(), v.end(), dst);
                } else {
                    const float inv = 1.f / static_cast<float>(counts[c]);
                    const float* src = new_cent.data() + c * dimension_;
                    for (std::size_t d = 0; d < dimension_; ++d) {
                        dst[d] = src[d] * inv;
                    }
                }
                if (metric_ == Metric::Cosine) {
                    normalize_centroid(c);
                }
            }
        }
    }

    void residual_into(const float* v, std::size_t list, float* out) const {
        const float* c = centroids_.data() + list * dimension_;
        for (std::size_t d = 0; d < dimension_; ++d) {
            out[d] = v[d] - c[d];
        }
    }

    void train_pq_codebooks(const std::vector<std::vector<float>>& training) {
        const std::size_t nt = training.size();
        const std::size_t ds = dsub();
        codebooks_.assign(params_.M * kSub * ds, 0.f);

        std::vector<float> residual(dimension_);
        // Per-subspace training matrix: nt * ds (reused)
        std::vector<float> sub_data(nt * ds);
        std::mt19937_64 rng(params_.seed + 1337);

        for (std::size_t m = 0; m < params_.M; ++m) {
            for (std::size_t i = 0; i < nt; ++i) {
                const std::size_t list = nearest_centroid(training[i].data());
                residual_into(training[i].data(), list, residual.data());
                std::copy(residual.begin() + static_cast<std::ptrdiff_t>(m * ds),
                          residual.begin() + static_cast<std::ptrdiff_t>((m + 1) * ds),
                          sub_data.begin() + static_cast<std::ptrdiff_t>(i * ds));
            }
            train_subspace_kmeans(m, sub_data, nt, ds, rng);
        }
    }

    void train_subspace_kmeans(std::size_t m, const std::vector<float>& sub_data, std::size_t nt,
                               std::size_t ds, std::mt19937_64& rng) {
        const std::size_t k = std::min(kSub, nt);
        // Init from random distinct rows
        std::vector<std::size_t> order(nt);
        for (std::size_t i = 0; i < nt; ++i) {
            order[i] = i;
        }
        for (std::size_t i = nt; i > 1; --i) {
            std::uniform_int_distribution<std::size_t> dist(0, i - 1);
            std::swap(order[i - 1], order[dist(rng)]);
        }
        for (std::size_t c = 0; c < k; ++c) {
            std::copy(sub_data.begin() + static_cast<std::ptrdiff_t>(order[c] * ds),
                      sub_data.begin() + static_cast<std::ptrdiff_t>((order[c] + 1) * ds),
                      codebook_entry(m, c));
        }
        // Remaining centroids (if k < kSub, rare tiny training) copy from first
        for (std::size_t c = k; c < kSub; ++c) {
            std::copy(codebook_entry(m, c % k), codebook_entry(m, c % k) + ds, codebook_entry(m, c));
        }

        std::vector<std::size_t> assign(nt, 0);
        std::vector<std::size_t> counts(kSub, 0);
        std::vector<float> new_cent(kSub * ds, 0.f);

        for (std::size_t iter = 0; iter < params_.pq_kmeans_iters; ++iter) {
            for (std::size_t i = 0; i < nt; ++i) {
                const float* x = sub_data.data() + i * ds;
                std::size_t best = 0;
                float best_d = simd::squared_l2(x, codebook_entry(m, 0), ds);
                for (std::size_t c = 1; c < kSub; ++c) {
                    const float d = simd::squared_l2(x, codebook_entry(m, c), ds);
                    if (d < best_d) {
                        best_d = d;
                        best = c;
                    }
                }
                assign[i] = best;
            }
            std::fill(new_cent.begin(), new_cent.end(), 0.f);
            std::fill(counts.begin(), counts.end(), 0);
            for (std::size_t i = 0; i < nt; ++i) {
                const std::size_t c = assign[i];
                ++counts[c];
                float* dst = new_cent.data() + c * ds;
                const float* src = sub_data.data() + i * ds;
                for (std::size_t d = 0; d < ds; ++d) {
                    dst[d] += src[d];
                }
            }
            for (std::size_t c = 0; c < kSub; ++c) {
                float* dst = codebook_entry(m, c);
                if (counts[c] == 0) {
                    std::uniform_int_distribution<std::size_t> dist(0, nt - 1);
                    const std::size_t j = dist(rng);
                    std::copy(sub_data.begin() + static_cast<std::ptrdiff_t>(j * ds),
                              sub_data.begin() + static_cast<std::ptrdiff_t>((j + 1) * ds), dst);
                } else {
                    const float inv = 1.f / static_cast<float>(counts[c]);
                    const float* src = new_cent.data() + c * ds;
                    for (std::size_t d = 0; d < ds; ++d) {
                        dst[d] = src[d] * inv;
                    }
                }
            }
        }
    }

    void encode_residual(const float* v, std::size_t list, std::uint8_t* code_out) const {
        std::vector<float> residual(dimension_);
        residual_into(v, list, residual.data());
        const std::size_t ds = dsub();
        for (std::size_t m = 0; m < params_.M; ++m) {
            const float* x = residual.data() + m * ds;
            std::size_t best = 0;
            float best_d = simd::squared_l2(x, codebook_entry(m, 0), ds);
            for (std::size_t c = 1; c < kSub; ++c) {
                const float d = simd::squared_l2(x, codebook_entry(m, c), ds);
                if (d < best_d) {
                    best_d = d;
                    best = c;
                }
            }
            code_out[m] = static_cast<std::uint8_t>(best);
        }
    }

    /// Base term independent of code for a given coarse list.
    /// Euclidean: 0 (all distance is in residual ADC).
    /// IP / Cosine: IP(q, coarse_centroid).
    float list_base_score(const float* q, std::size_t list) const {
        if (metric_ == Metric::Euclidean) {
            return 0.f;
        }
        return inner_product(q, centroids_.data() + list * dimension_, dimension_);
    }

    void build_adc_table(const float* q, std::size_t list, float* residual_q,
                         float* table) const {
        residual_into(q, list, residual_q);
        const std::size_t ds = dsub();
        if (metric_ == Metric::Euclidean) {
            // table[m][k] = || r_q_m - codebook[m][k] ||^2
            for (std::size_t m = 0; m < params_.M; ++m) {
                const float* rq = residual_q + m * ds;
                for (std::size_t c = 0; c < kSub; ++c) {
                    table[m * kSub + c] = simd::squared_l2(rq, codebook_entry(m, c), ds);
                }
            }
        } else {
            // IP / Cosine: table[m][k] = IP(q_m, codebook[m][k]) on residual subspace.
            // Full score = IP(q, centroid) + sum_m table[m][code]  (since x≈c+decode(r)).
            // Use full query subspace (not residual query) for IP with residual codes:
            // IP(q, c + r_hat) = IP(q,c) + IP(q, r_hat); r_hat from codebook on residual.
            for (std::size_t m = 0; m < params_.M; ++m) {
                const float* qm = q + m * ds;
                for (std::size_t c = 0; c < kSub; ++c) {
                    table[m * kSub + c] = inner_product(qm, codebook_entry(m, c), ds);
                }
            }
        }
    }

    float adc_from_table(const std::uint8_t* code, const float* table) const {
        float s = 0.f;
        for (std::size_t m = 0; m < params_.M; ++m) {
            s += table[m * kSub + code[m]];
        }
        if (metric_ == Metric::Euclidean) {
            // squared L2 residual distance (order-preserving for ranking)
            return s;
        }
        // IP / Cosine: higher is better (sum of IP contributions)
        return s;
    }

    void remove_node_at(int idx) {
        const int n = static_cast<int>(ids_.size());
        if (idx < 0 || idx >= n) {
            return;
        }
        const int last = n - 1;
        const std::size_t list = list_of_[static_cast<std::size_t>(idx)];
        auto& lst = lists_[list];
        lst.erase(std::remove(lst.begin(), lst.end(), idx), lst.end());

        if (idx != last) {
            const std::size_t last_list = list_of_[static_cast<std::size_t>(last)];
            auto& ll = lists_[last_list];
            ll.erase(std::remove(ll.begin(), ll.end(), last), ll.end());

            ids_[static_cast<std::size_t>(idx)] = ids_[static_cast<std::size_t>(last)];
            list_of_[static_cast<std::size_t>(idx)] = list_of_[static_cast<std::size_t>(last)];
            std::copy(codes_.begin() + static_cast<std::ptrdiff_t>(last * static_cast<int>(params_.M)),
                      codes_.begin() + static_cast<std::ptrdiff_t>((last + 1) * static_cast<int>(params_.M)),
                      codes_.begin() + static_cast<std::ptrdiff_t>(idx * static_cast<int>(params_.M)));
            lists_[list_of_[static_cast<std::size_t>(idx)]].push_back(idx);
        }

        ids_.pop_back();
        list_of_.pop_back();
        codes_.resize(ids_.size() * params_.M);
    }

    void validate_loaded() const {
        const std::size_t n = ids_.size();
        if (list_of_.size() != n) {
            throw std::runtime_error("tinyann::IvfPqIndex::load: list_of size mismatch");
        }
        if (codes_.size() != n * params_.M) {
            throw std::runtime_error("tinyann::IvfPqIndex::load: codes size mismatch");
        }
        if (centroids_.size() != params_.nlist * dimension_) {
            throw std::runtime_error("tinyann::IvfPqIndex::load: centroids size mismatch");
        }
        if (codebooks_.size() != params_.M * kSub * dsub()) {
            throw std::runtime_error("tinyann::IvfPqIndex::load: codebooks size mismatch");
        }
        if (lists_.size() != params_.nlist) {
            throw std::runtime_error("tinyann::IvfPqIndex::load: lists size mismatch");
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (list_of_[i] >= params_.nlist) {
                throw std::runtime_error("tinyann::IvfPqIndex::load: list_of out of range");
            }
        }
        std::vector<std::uint8_t> seen(n, 0);
        for (std::size_t c = 0; c < params_.nlist; ++c) {
            for (int node : lists_[c]) {
                if (node < 0 || static_cast<std::size_t>(node) >= n) {
                    throw std::runtime_error("tinyann::IvfPqIndex::load: list node OOB");
                }
                const std::size_t u = static_cast<std::size_t>(node);
                if (list_of_[u] != c) {
                    throw std::runtime_error("tinyann::IvfPqIndex::load: lists disagree list_of");
                }
                if (seen[u] != 0) {
                    throw std::runtime_error("tinyann::IvfPqIndex::load: duplicate list node");
                }
                seen[u] = 1;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (seen[i] == 0) {
                throw std::runtime_error("tinyann::IvfPqIndex::load: node missing from lists");
            }
        }
    }

    void rebuild_lists_from_list_of() {
        lists_.assign(params_.nlist, {});
        for (std::size_t i = 0; i < list_of_.size(); ++i) {
            lists_[list_of_[i]].push_back(static_cast<int>(i));
        }
    }

    std::size_t dimension_;
    Metric metric_;
    IvfPqParams params_;
    bool trained_ = false;

    std::vector<float> centroids_;   // nlist * dim
    std::vector<float> codebooks_;   // M * kSub * dsub
    std::vector<std::vector<int>> lists_;
    std::vector<std::int64_t> ids_;
    std::vector<std::size_t> list_of_;
    std::vector<std::uint8_t> codes_;  // n * M
};

}  // namespace tinyann
