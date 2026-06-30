#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tinyann {

/// Distance / similarity metrics supported by the index.
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

/// In-memory exact (brute-force) vector similarity index.
///
/// Vectors have a fixed dimension set at construction. `add` stores (id, vector)
/// pairs; `search` returns the k best matches under the configured metric.
///
/// Ranking:
/// - Cosine and InnerProduct are similarities (higher score = better/closer)
/// - Euclidean is a distance (lower score = better/closer)
class Index {
public:
    /// Construct an empty index for vectors of the given dimension and metric.
    /// @throws std::invalid_argument if dimension == 0
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

    /// Number of dimensions for every vector in this index.
    std::size_t dimension() const noexcept { return dimension_; }

    /// Metric used for scoring and ranking.
    Metric metric() const noexcept { return metric_; }

    /// Number of vectors currently stored.
    std::size_t size() const noexcept { return ids_.size(); }

    /// Whether the index has no vectors.
    bool empty() const noexcept { return ids_.empty(); }

    /// Insert a vector with the given id.
    /// Replacing an existing id is not supported; duplicate ids are allowed
    /// (both entries are kept and may appear in search results).
    /// @throws std::invalid_argument if vector.size() != dimension()
    void add(std::int64_t id, const std::vector<float>& vector) {
        if (vector.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::add: vector dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(vector.size()) + ")");
        }
        ids_.push_back(id);
        data_.insert(data_.end(), vector.begin(), vector.end());
    }

    /// Exact k-NN search.
    /// Returns up to k results ranked best-first for the index metric.
    /// If k is 0 or the index is empty, returns an empty vector.
    /// If k > size(), returns all vectors (size() results).
    /// Zero-norm vectors under Cosine yield score 0 (undefined cosine treated as 0).
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        if (query.size() != dimension_) {
            throw std::invalid_argument(
                "tinyann::Index::search: query dimension mismatch (expected " +
                std::to_string(dimension_) + ", got " + std::to_string(query.size()) + ")");
        }
        if (k == 0 || empty()) {
            return {};
        }

        const std::size_t n = size();
        const std::size_t take = std::min(k, n);

        std::vector<SearchResult> all;
        all.reserve(n);

        const float* base = data_.data();
        for (std::size_t i = 0; i < n; ++i) {
            const float* vec = base + i * dimension_;
            const float score = score_pair(query.data(), vec);
            all.push_back(SearchResult{ids_[i], score});
        }

        const bool higher_is_better = (metric_ != Metric::Euclidean);

        // Partial sort so the first `take` elements are the best, in ranked order.
        auto better = [higher_is_better](const SearchResult& a, const SearchResult& b) {
            if (a.score != b.score) {
                return higher_is_better ? (a.score > b.score) : (a.score < b.score);
            }
            // Stable tie-break on id for deterministic output
            return a.id < b.id;
        };

        if (take < n) {
            std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(take),
                              all.end(), better);
            all.resize(take);
        } else {
            std::sort(all.begin(), all.end(), better);
        }

        return all;
    }

    /// Compute the metric score between two vectors of this index's dimension.
    /// Exposed for testing and CLI diagnostics.
    float score(const std::vector<float>& a, const std::vector<float>& b) const {
        if (a.size() != dimension_ || b.size() != dimension_) {
            throw std::invalid_argument("tinyann::Index::score: dimension mismatch");
        }
        return score_pair(a.data(), b.data());
    }

    /// Human-readable name of a metric.
    static const char* metric_name(Metric m) {
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

    /// Parse metric from a string ("cosine", "euclidean"/"l2", "inner_product"/"ip").
    /// @throws std::invalid_argument on unknown name
    static Metric parse_metric(const std::string& name) {
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

private:
    float score_pair(const float* a, const float* b) const {
        switch (metric_) {
            case Metric::Cosine:
                return cosine_similarity(a, b, dimension_);
            case Metric::Euclidean:
                return euclidean_distance(a, b, dimension_);
            case Metric::InnerProduct:
                return inner_product(a, b, dimension_);
        }
        return 0.f;
    }

    static float inner_product(const float* a, const float* b, std::size_t dim) {
        float sum = 0.f;
        for (std::size_t i = 0; i < dim; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static float euclidean_distance(const float* a, const float* b, std::size_t dim) {
        float sum = 0.f;
        for (std::size_t i = 0; i < dim; ++i) {
            const float d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    /// Cosine similarity. Zero-norm on either side yields 0 (safe default).
    static float cosine_similarity(const float* a, const float* b, std::size_t dim) {
        float dot = 0.f;
        float na = 0.f;
        float nb = 0.f;
        for (std::size_t i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0.f || nb == 0.f) {
            return 0.f;
        }
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    std::size_t dimension_;
    Metric metric_;
    std::vector<std::int64_t> ids_;
    std::vector<float> data_;  // row-major: size() * dimension_ floats
};

}  // namespace tinyann
