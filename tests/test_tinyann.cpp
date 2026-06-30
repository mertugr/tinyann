#include "tinyann/tinyann.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  "      \
                      << #cond << "\n";                                      \
            ++g_failed;                                                      \
        } else {                                                             \
            ++g_passed;                                                      \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                \
    do {                                                                     \
        const double _a = static_cast<double>(a);                            \
        const double _b = static_cast<double>(b);                            \
        if (std::fabs(_a - _b) > (eps)) {                                    \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__              \
                      << "  near(" << _a << ", " << _b << ", eps=" << (eps)  \
                      << ")\n";                                              \
            ++g_failed;                                                      \
        } else {                                                             \
            ++g_passed;                                                      \
        }                                                                    \
    } while (0)

void test_construct_rejects_zero_dim() {
    bool threw = false;
    try {
        tinyann::Index idx(0, tinyann::Metric::Cosine);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_add_rejects_wrong_dim() {
    tinyann::Index idx(3, tinyann::Metric::Cosine);
    bool threw = false;
    try {
        idx.add(1, {1.f, 2.f});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(idx.empty());
    CHECK(idx.size() == 0);
}

void test_search_rejects_wrong_dim() {
    tinyann::Index idx(2, tinyann::Metric::Cosine);
    idx.add(1, {1.f, 0.f});
    bool threw = false;
    try {
        (void)idx.search({1.f}, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_empty_index_search() {
    tinyann::Index idx(2, tinyann::Metric::Cosine);
    auto r = idx.search({1.f, 0.f}, 5);
    CHECK(r.empty());
    CHECK(idx.empty());
}

void test_k_zero_returns_empty() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    idx.add(1, {1.f, 0.f});
    auto r = idx.search({1.f, 0.f}, 0);
    CHECK(r.empty());
}

void test_k_greater_than_size() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    idx.add(10, {1.f, 0.f});
    idx.add(20, {0.f, 1.f});
    auto r = idx.search({1.f, 0.f}, 100);
    CHECK(r.size() == 2);
    CHECK(r[0].id == 10);
    CHECK(r[1].id == 20);
}

void test_inner_product_ranking() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    idx.add(1, {1.f, 0.f});
    idx.add(2, {0.5f, 0.f});
    idx.add(3, {0.f, 1.f});
    idx.add(4, {-1.f, 0.f});
    auto r = idx.search({1.f, 0.f}, 3);
    CHECK(r.size() == 3);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 1.0, 1e-5);
    CHECK(r[1].id == 2);
    CHECK_NEAR(r[1].score, 0.5, 1e-5);
    CHECK(r[2].id == 3);
    CHECK_NEAR(r[2].score, 0.0, 1e-5);
}

void test_euclidean_ranking() {
    tinyann::Index idx(2, tinyann::Metric::Euclidean);
    idx.add(1, {1.f, 0.f});
    idx.add(2, {1.f, 1.f});
    idx.add(3, {0.f, 0.f});
    idx.add(4, {4.f, 0.f});
    auto r = idx.search({1.f, 0.f}, 3);
    CHECK(r.size() == 3);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 0.0, 1e-5);
    CHECK(r[1].id == 2);
    CHECK_NEAR(r[1].score, 1.0, 1e-5);
    CHECK(r[2].id == 3);
    CHECK_NEAR(r[2].score, 1.0, 1e-5);
}

void test_cosine_ranking() {
    tinyann::Index idx(2, tinyann::Metric::Cosine);
    idx.add(1, {2.f, 0.f});
    idx.add(2, {1.f, 1.f});
    idx.add(3, {0.f, 1.f});
    idx.add(4, {-1.f, 0.f});
    auto r = idx.search({1.f, 0.f}, 4);
    CHECK(r.size() == 4);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 1.0, 1e-5);
    CHECK(r[1].id == 2);
    CHECK_NEAR(r[1].score, 1.0 / std::sqrt(2.0), 1e-5);
    CHECK(r[2].id == 3);
    CHECK_NEAR(r[2].score, 0.0, 1e-5);
    CHECK(r[3].id == 4);
    CHECK_NEAR(r[3].score, -1.0, 1e-5);
}

void test_cosine_zero_vector() {
    tinyann::Index idx(3, tinyann::Metric::Cosine);
    idx.add(1, {0.f, 0.f, 0.f});
    idx.add(2, {1.f, 0.f, 0.f});
    auto r0 = idx.search({0.f, 0.f, 0.f}, 2);
    CHECK(r0.size() == 2);
    CHECK_NEAR(r0[0].score, 0.0, 1e-6);
    CHECK_NEAR(r0[1].score, 0.0, 1e-6);

    auto r1 = idx.search({1.f, 0.f, 0.f}, 2);
    CHECK(r1.size() == 2);
    CHECK(r1[0].id == 2);
    CHECK_NEAR(r1[0].score, 1.0, 1e-5);
    CHECK(r1[1].id == 1);
    CHECK_NEAR(r1[1].score, 0.0, 1e-6);
}

void test_euclidean_zero_vectors() {
    tinyann::Index idx(2, tinyann::Metric::Euclidean);
    idx.add(1, {0.f, 0.f});
    idx.add(2, {3.f, 4.f});
    auto r = idx.search({0.f, 0.f}, 2);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 0.0, 1e-6);
    CHECK(r[1].id == 2);
    CHECK_NEAR(r[1].score, 5.0, 1e-5);
}

void test_inner_product_zero_vectors() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    idx.add(1, {0.f, 0.f});
    idx.add(2, {1.f, 2.f});
    auto r = idx.search({0.f, 0.f}, 2);
    CHECK(r.size() == 2);
    CHECK_NEAR(r[0].score, 0.0, 1e-6);
    CHECK_NEAR(r[1].score, 0.0, 1e-6);
    CHECK(r[0].id == 1);
    CHECK(r[1].id == 2);
}

void test_score_helpers() {
    tinyann::Index cos(2, tinyann::Metric::Cosine);
    CHECK_NEAR(cos.score({1.f, 0.f}, {0.f, 1.f}), 0.0, 1e-6);
    CHECK_NEAR(cos.score({1.f, 0.f}, {1.f, 0.f}), 1.0, 1e-6);

    tinyann::Index l2(2, tinyann::Metric::Euclidean);
    CHECK_NEAR(l2.score({0.f, 0.f}, {3.f, 4.f}), 5.0, 1e-5);

    tinyann::Index ip(2, tinyann::Metric::InnerProduct);
    CHECK_NEAR(ip.score({1.f, 2.f}, {3.f, 4.f}), 11.0, 1e-5);
}

void test_parse_metric() {
    CHECK(tinyann::Index::parse_metric("cosine") == tinyann::Metric::Cosine);
    CHECK(tinyann::Index::parse_metric("cos") == tinyann::Metric::Cosine);
    CHECK(tinyann::Index::parse_metric("euclidean") == tinyann::Metric::Euclidean);
    CHECK(tinyann::Index::parse_metric("l2") == tinyann::Metric::Euclidean);
    CHECK(tinyann::Index::parse_metric("inner_product") == tinyann::Metric::InnerProduct);
    CHECK(tinyann::Index::parse_metric("ip") == tinyann::Metric::InnerProduct);
    bool threw = false;
    try {
        (void)tinyann::Index::parse_metric("nope");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_metric_name() {
    CHECK(std::string(tinyann::Index::metric_name(tinyann::Metric::Cosine)) == "cosine");
    CHECK(std::string(tinyann::Index::metric_name(tinyann::Metric::Euclidean)) == "euclidean");
    CHECK(std::string(tinyann::Index::metric_name(tinyann::Metric::InnerProduct)) ==
          "inner_product");
}

void test_duplicate_ids_kept() {
    tinyann::Index idx(1, tinyann::Metric::InnerProduct);
    idx.add(7, {1.f});
    idx.add(7, {2.f});
    CHECK(idx.size() == 2);
    auto r = idx.search({1.f}, 2);
    CHECK(r.size() == 2);
    CHECK(r[0].id == 7);
    CHECK_NEAR(r[0].score, 2.0, 1e-5);
    CHECK(r[1].id == 7);
    CHECK_NEAR(r[1].score, 1.0, 1e-5);
}

void test_k_equals_one() {
    tinyann::Index idx(3, tinyann::Metric::Cosine);
    idx.add(100, {1.f, 0.f, 0.f});
    idx.add(200, {0.f, 1.f, 0.f});
    idx.add(300, {0.9f, 0.1f, 0.f});
    auto r = idx.search({1.f, 0.f, 0.f}, 1);
    CHECK(r.size() == 1);
    CHECK(r[0].id == 100);
}

void test_accessors() {
    tinyann::Index idx(4, tinyann::Metric::Euclidean);
    CHECK(idx.dimension() == 4);
    CHECK(idx.metric() == tinyann::Metric::Euclidean);
    idx.add(1, {1.f, 2.f, 3.f, 4.f});
    CHECK(idx.size() == 1);
    CHECK(!idx.empty());
}

void test_high_dim_inner_product() {
    const std::size_t dim = 128;
    tinyann::Index idx(dim, tinyann::Metric::InnerProduct);
    std::vector<float> a(dim, 0.f);
    std::vector<float> b(dim, 0.f);
    std::vector<float> q(dim, 0.f);
    a[0] = 1.f;
    b[1] = 1.f;
    q[0] = 2.f;
    idx.add(1, a);
    idx.add(2, b);
    auto r = idx.search(q, 2);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 2.0, 1e-5);
    CHECK(r[1].id == 2);
    CHECK_NEAR(r[1].score, 0.0, 1e-5);
}

void test_negative_ids() {
    tinyann::Index idx(1, tinyann::Metric::InnerProduct);
    idx.add(-5, {3.f});
    idx.add(0, {1.f});
    auto r = idx.search({1.f}, 2);
    CHECK(r[0].id == -5);
    CHECK_NEAR(r[0].score, 3.0, 1e-5);
    CHECK(r[1].id == 0);
}

void test_recall_helper() {
    std::vector<tinyann::SearchResult> exact = {{1, 0.f}, {2, 0.f}, {3, 0.f}};
    std::vector<tinyann::SearchResult> approx = {{2, 0.f}, {9, 0.f}, {1, 0.f}};
    CHECK_NEAR(tinyann::recall_at_k(approx, exact), 2.0 / 3.0, 1e-9);
    CHECK_NEAR(tinyann::recall_at_k({}, {}), 1.0, 1e-9);
    CHECK_NEAR(tinyann::recall_at_k(approx, {}), 1.0, 1e-9);
}

void test_hnsw_empty_and_k0() {
    tinyann::HnswIndex h(2, tinyann::Metric::Cosine);
    CHECK(h.search({1.f, 0.f}, 5).empty());
    h.add(1, {1.f, 0.f});
    CHECK(h.search({1.f, 0.f}, 0).empty());
}

void test_hnsw_small_exact_match_ip() {
    // On tiny graphs with high ef, HNSW should recover the exact top-1.
    tinyann::HnswParams p;
    p.M = 8;
    p.ef_construction = 64;
    p.ef_search = 32;
    p.seed = 1;
    tinyann::HnswIndex h(2, tinyann::Metric::InnerProduct, p);
    h.add(1, {1.f, 0.f});
    h.add(2, {0.f, 1.f});
    h.add(3, {0.5f, 0.f});
    auto r = h.search({1.f, 0.f}, 1);
    CHECK(r.size() == 1);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 1.0, 1e-5);
}

void test_hnsw_rejects_bad_dim() {
    tinyann::HnswIndex h(3, tinyann::Metric::Euclidean);
    bool threw = false;
    try {
        h.add(1, {1.f, 2.f});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

std::vector<float> random_unit_vector(std::size_t dim, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(dim);
    float norm2 = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] = nd(rng);
        norm2 += v[i] * v[i];
    }
    if (norm2 > 0.f) {
        const float inv = 1.f / std::sqrt(norm2);
        for (float& x : v) {
            x *= inv;
        }
    }
    return v;
}

void assert_hnsw_recall(tinyann::Metric metric, const char* label) {
    const std::size_t dim = 32;
    const std::size_t n = 2000;
    const std::size_t nq = 50;
    const std::size_t k = 10;

    tinyann::HnswParams p;
    p.M = 16;
    p.ef_construction = 200;
    p.ef_search = 64;
    p.seed = 12345;

    tinyann::Index exact(dim, metric);
    tinyann::HnswIndex hnsw(dim, metric, p);

    std::mt19937_64 rng(999);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = random_unit_vector(dim, rng);
        const std::int64_t id = static_cast<std::int64_t>(i);
        exact.add(id, v);
        hnsw.add(id, v);
    }

    std::vector<std::vector<float>> queries;
    queries.reserve(nq);
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }

    const double rec = hnsw.recall_at_k_vs(exact, queries, k, /*ef=*/64);
    std::cout << "recall@10[" << label << "]=" << rec << " (n=" << n << " dim=" << dim
              << " nq=" << nq << ")\n";
    CHECK(rec > 0.9);
}

void test_hnsw_recall_cosine() { assert_hnsw_recall(tinyann::Metric::Cosine, "cosine"); }
void test_hnsw_recall_euclidean() { assert_hnsw_recall(tinyann::Metric::Euclidean, "euclidean"); }
void test_hnsw_recall_inner_product() {
    assert_hnsw_recall(tinyann::Metric::InnerProduct, "inner_product");
}

}  // namespace

int main() {
    test_construct_rejects_zero_dim();
    test_add_rejects_wrong_dim();
    test_search_rejects_wrong_dim();
    test_empty_index_search();
    test_k_zero_returns_empty();
    test_k_greater_than_size();
    test_inner_product_ranking();
    test_euclidean_ranking();
    test_cosine_ranking();
    test_cosine_zero_vector();
    test_euclidean_zero_vectors();
    test_inner_product_zero_vectors();
    test_score_helpers();
    test_parse_metric();
    test_metric_name();
    test_duplicate_ids_kept();
    test_k_equals_one();
    test_accessors();
    test_high_dim_inner_product();
    test_negative_ids();
    test_recall_helper();
    test_hnsw_empty_and_k0();
    test_hnsw_small_exact_match_ip();
    test_hnsw_rejects_bad_dim();
    test_hnsw_recall_cosine();
    test_hnsw_recall_euclidean();
    test_hnsw_recall_inner_product();

    std::cout << "passed=" << g_passed << " failed=" << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
