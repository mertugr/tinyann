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
/// Cosine → 1 - sim; InnerProduct → -dot; Euclidean → L2.
inline float ann_distance(Metric m, const float* a, const float* b, std::size_t dim) {
    switch (m) {
        case Metric::Cosine:
            return 1.f - cosine_similarity(a, b, dim);
        case Metric::Euclidean:
            return euclidean_distance(a, b, dim);
        case Metric::InnerProduct:
            return -inner_product(a, b, dim);
    }
    return 0.f;
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
// Binary persistence helpers (little-endian host; fixed-width types)
// Format: magic "TANN" | version u32 | kind u32 | ... payload ...
// kind: 1 = Exact Index, 2 = HnswIndex
// ---------------------------------------------------------------------------

namespace detail {

constexpr char kMagic[4] = {'T', 'A', 'N', 'N'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kKindExact = 1;
constexpr std::uint32_t kKindHnsw = 2;

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
    dimension = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    const auto count = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    if (dimension == 0) {
        throw std::runtime_error("tinyann: corrupt file (dimension == 0)");
    }
    ids.resize(count);
    data.resize(count * dimension);
    if (count > 0) {
        read_bytes(in, ids.data(), count * sizeof(std::int64_t));
        read_bytes(in, data.data(), count * dimension * sizeof(float));
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Exact (brute-force) index
// ---------------------------------------------------------------------------

/// In-memory exact (brute-force) vector similarity index.
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

    void add(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
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
    /// @throws std::invalid_argument on dimension mismatch
    bool update(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::update: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
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

    void add(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::HnswIndex::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }

        const int node = static_cast<int>(ids_.size());
        ids_.push_back(id);
        data_.insert(data_.end(), vector.begin(), vector.end());

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

    /// True if at least one node has this id.
    bool contains(std::int64_t id) const {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

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
    template <typename Pred>
    auto search(const std::vector<float>& query, std::size_t k, Pred predicate) const
        -> std::enable_if_t<std::is_invocable_r<bool, Pred&, std::int64_t>::value,
                            std::vector<SearchResult>> {
        return search(query, k, params_.ef_search, std::move(predicate));
    }

    /// Filtered approximate k-NN with explicit ef (exploration width over eligible hits).
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
        if (k == 0 || empty()) {
            return {};
        }
        if (ef == 0) {
            throw std::invalid_argument("ef must be > 0");
        }
        ef = std::max(ef, k);

        int curr = entry_point_;
        for (int lc = max_level_; lc > 0; --lc) {
            // Upper layers: navigate on the full graph (no filter) for connectivity.
            curr = greedy_update_query(query.data(), curr, lc);
        }

        auto candidates =
            search_layer_query_filtered(query.data(), curr, ef, /*layer=*/0, predicate);
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
        params.M = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(in));
        params.ef_construction = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(in));
        params.ef_search = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(in));
        params.seed = detail::read_pod<std::uint64_t>(in);

        HnswIndex idx(dimension, metric, params);
        idx.ids_ = std::move(ids);
        idx.data_ = std::move(data);
        idx.entry_point_ = detail::read_pod<std::int32_t>(in);
        idx.max_level_ = detail::read_pod<std::int32_t>(in);

        const std::size_t n = idx.ids_.size();
        idx.levels_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            idx.levels_[i] = detail::read_pod<std::int32_t>(in);
        }
        idx.neighbors_.assign(n, {});
        for (std::size_t i = 0; i < n; ++i) {
            const auto n_layers = detail::read_pod<std::uint32_t>(in);
            idx.neighbors_[i].resize(n_layers);
            for (std::uint32_t lc = 0; lc < n_layers; ++lc) {
                const auto n_links = detail::read_pod<std::uint32_t>(in);
                idx.neighbors_[i][lc].resize(n_links);
                for (std::uint32_t j = 0; j < n_links; ++j) {
                    idx.neighbors_[i][lc][j] = detail::read_pod<std::int32_t>(in);
                }
            }
        }

        const auto rng_len = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(in));
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

    float distance_nodes(int a, int b) const {
        const float* pa = data_.data() + static_cast<std::size_t>(a) * dimension_;
        const float* pb = data_.data() + static_cast<std::size_t>(b) * dimension_;
        return ann_distance(metric_, pa, pb, dimension_);
    }

    float distance_query(const float* q, int node) const {
        const float* p = data_.data() + static_cast<std::size_t>(node) * dimension_;
        return ann_distance(metric_, q, p, dimension_);
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

    int greedy_update_query(const float* q, int enter, int layer) const {
        int curr = enter;
        float d = distance_query(q, curr);
        bool changed = true;
        while (changed) {
            changed = false;
            const auto& links =
                neighbors_[static_cast<std::size_t>(curr)][static_cast<std::size_t>(layer)];
            for (int nb : links) {
                const float dn = distance_query(q, nb);
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
        return search_layer_query_filtered(q, enter, ef, layer, [](std::int64_t) { return true; });
    }

    /// Layer search with id filter applied to the *eligible result* set only.
    /// Graph edges are still followed through ineligible nodes; those nodes never
    /// enter the result heap — unlike post-filtering an unfiltered top-ef list.
    template <typename Pred>
    std::vector<DistNode> search_layer_query_filtered(const float* q, int enter, std::size_t ef,
                                                      int layer, Pred predicate) const {
        std::unordered_set<int> visited;
        visited.reserve(ef * 8);

        // Exploration frontier (all nodes, for connectivity).
        std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates;
        // Eligible dynamic list only (max-heap, size <= ef).
        std::priority_queue<DistNode> eligible;

        const float d0 = distance_query(q, enter);
        candidates.emplace(d0, enter);
        visited.insert(enter);
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
                if (!visited.insert(nb).second) {
                    continue;
                }
                const float d = distance_query(q, nb);
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
    mutable std::mt19937_64 rng_;
    double level_mult_;

    std::vector<std::int64_t> ids_;
    std::vector<float> data_;
    std::vector<int> levels_;
    std::vector<std::vector<std::vector<int>>> neighbors_;
    int entry_point_ = -1;
    int max_level_ = -1;
};

}  // namespace tinyann
