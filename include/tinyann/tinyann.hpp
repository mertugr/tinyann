#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tinyann {

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
// Shared metric helpers (used by exact Index and HnswIndex)
// ---------------------------------------------------------------------------

inline bool higher_is_better(Metric m) noexcept { return m != Metric::Euclidean; }

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    float sum = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float euclidean_distance(const float* a, const float* b, std::size_t dim) {
    float sum = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

/// Cosine similarity. Zero-norm on either side yields 0 (safe default).
inline float cosine_similarity(const float* a, const float* b, std::size_t dim) {
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
///
/// Defined as |approx_ids ∩ exact_ids| / |exact_ids|, where both lists are the
/// top-k (or fewer if the index is smaller) results for the same query.
/// Duplicate ids in either list are treated as a set (each id counted once).
/// Returns 1.0 when exact is empty (vacuously perfect).
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
// Exact (brute-force) index
// ---------------------------------------------------------------------------

/// In-memory exact (brute-force) vector similarity index.
///
/// Ranking:
/// - Cosine and InnerProduct are similarities (higher score = better/closer)
/// - Euclidean is a distance (lower score = better/closer)
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

    /// Exact k-NN search. Best-first; empty if k==0 or index empty; all if k>n.
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
            all.push_back(SearchResult{
                ids_[i], metric_score(metric_, query.data(), base + i * dimension_, dimension_)});
        }

        const bool hib = higher_is_better(metric_);
        auto better = [hib](const SearchResult& a, const SearchResult& b) {
            if (a.score != b.score) {
                return hib ? (a.score > b.score) : (a.score < b.score);
            }
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

    float score(const std::vector<float>& a, const std::vector<float>& b) const {
        if (a.size() != dimension_ || b.size() != dimension_) {
            throw std::invalid_argument("tinyann::Index::score: dimension mismatch");
        }
        return metric_score(metric_, a.data(), b.data(), dimension_);
    }

    static const char* metric_name(Metric m) { return tinyann::metric_name(m); }
    static Metric parse_metric(const std::string& name) { return tinyann::parse_metric(name); }

    /// Access underlying storage (for building ANN from the same data, tests).
    const std::vector<std::int64_t>& ids() const noexcept { return ids_; }
    const std::vector<float>& data() const noexcept { return data_; }

private:
    std::size_t dimension_;
    Metric metric_;
    std::vector<std::int64_t> ids_;
    std::vector<float> data_;
};

// ---------------------------------------------------------------------------
// Approximate index: HNSW (Hierarchical Navigable Small World)
//
// Chosen over IVF because:
// - Incremental inserts with no separate k-means training phase (matches add())
// - Strong in-memory recall/latency tradeoff for general k-NN workloads
// - Tunable at query time via ef_search without rebuilding clusters
// - Modern default for in-process ANN libraries (e.g. hnswlib, usearch)
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
/// Same metrics and SearchResult ranking semantics as Index; search may miss
/// some true neighbors in exchange for sub-linear query cost on larger sets.
class HnswIndex {
public:
    explicit HnswIndex(std::size_t dimension, Metric metric = Metric::Cosine,
                       HnswParams params = {})
        : dimension_(dimension),
          metric_(metric),
          params_(params),
          rng_(params.seed),
          // level mult ≈ 1/ln(M); clamp M to avoid ln(1)
          level_mult_(1.0 / std::log(std::max<std::size_t>(params.M, 2))) {
        if (dimension_ == 0) {
            throw std::invalid_argument("tinyann::HnswIndex: dimension must be > 0");
        }
        if (params_.M == 0) {
            throw std::invalid_argument("tinyann::HnswIndex: M must be > 0");
        }
        if (params_.ef_construction == 0 || params_.ef_search == 0) {
            throw std::invalid_argument("tinyann::HnswIndex: ef_construction/ef_search must be > 0");
        }
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
        // neighbors_[node][layer] = list of neighbor node indices
        neighbors_.emplace_back();
        neighbors_.back().resize(static_cast<std::size_t>(level) + 1);

        if (node == 0) {
            entry_point_ = 0;
            max_level_ = level;
            return;
        }

        int curr = entry_point_;
        // Greedy search from top layer down to level+1
        for (int lc = max_level_; lc > level; --lc) {
            curr = greedy_update(node, curr, lc);
        }

        // Insert on layers level .. 0
        for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
            auto candidates = search_layer(node, /*query_is_node=*/true, curr, params_.ef_construction, lc);
            const std::size_t M_max = (lc == 0) ? (params_.M * 2) : params_.M;
            auto selected = select_neighbors(candidates, M_max);

            neighbors_[static_cast<std::size_t>(node)][static_cast<std::size_t>(lc)] = selected;

            // Add reverse edges and prune
            for (int nb : selected) {
                auto& nb_links = neighbors_[static_cast<std::size_t>(nb)][static_cast<std::size_t>(lc)];
                nb_links.push_back(node);
                if (nb_links.size() > M_max) {
                    // Prune neighbor's links to M_max closest to nb
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
                // Next layer enters from the closest candidate
                curr = candidates[0].second;
            }
        }

        if (level > max_level_) {
            entry_point_ = node;
            max_level_ = level;
        }
    }

    /// Approximate k-NN search. Uses params().ef_search (must be >= k for best recall).
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const {
        return search(query, k, params_.ef_search);
    }

    /// Approximate k-NN with explicit ef (candidate list width). Larger ef → higher recall.
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k,
                                     std::size_t ef) const {
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
            curr = greedy_update_query(query.data(), curr, lc);
        }

        auto candidates = search_layer_query(query.data(), curr, ef, /*layer=*/0);
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

    /// Mean recall@k of this index vs an exact Index on the given queries.
    /// Both indexes must use the same metric and contain the same vectors/ids
    /// for the measure to be meaningful.
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

private:
    // Priority queue element: (distance, node); min-heap via greater
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
        // floor(-ln(U) * level_mult) — standard HNSW level sampling
        const double u = std::max(dist(rng_), 1e-12);
        const int lvl = static_cast<int>(-std::log(u) * level_mult_);
        return lvl;
    }

    /// Greedy descent toward node `target` on a single layer (used during insert).
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

    /// Search layer returning up to `ef` closest nodes as (dist, node), sorted ascending dist.
    /// When query_is_node, query vector is taken from data_[query_node].
    std::vector<DistNode> search_layer(int query_node, bool /*query_is_node*/, int enter,
                                       std::size_t ef, int layer) const {
        const float* q = data_.data() + static_cast<std::size_t>(query_node) * dimension_;
        return search_layer_query(q, enter, ef, layer);
    }

    std::vector<DistNode> search_layer_query(const float* q, int enter, std::size_t ef,
                                             int layer) const {
        std::unordered_set<int> visited;
        visited.reserve(ef * 4);

        // candidates: min-heap (closest first)
        std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates;
        // results: max-heap of size <= ef (farthest of current best on top)
        std::priority_queue<DistNode> results;

        const float d0 = distance_query(q, enter);
        candidates.emplace(d0, enter);
        results.emplace(d0, enter);
        visited.insert(enter);

        while (!candidates.empty()) {
            const DistNode c = candidates.top();
            candidates.pop();
            const float farthest = results.top().first;
            if (c.first > farthest) {
                break;
            }
            const auto& links =
                neighbors_[static_cast<std::size_t>(c.second)][static_cast<std::size_t>(layer)];
            for (int nb : links) {
                if (!visited.insert(nb).second) {
                    continue;
                }
                const float d = distance_query(q, nb);
                if (results.size() < ef || d < results.top().first) {
                    candidates.emplace(d, nb);
                    results.emplace(d, nb);
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }

        std::vector<DistNode> out;
        out.reserve(results.size());
        while (!results.empty()) {
            out.push_back(results.top());
            results.pop();
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
    // neighbors_[node][layer] -> neighbor node indices
    std::vector<std::vector<std::vector<int>>> neighbors_;
    int entry_point_ = -1;
    int max_level_ = -1;
};

}  // namespace tinyann
