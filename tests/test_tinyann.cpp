#include "tinyann/tinyann.hpp"
#include "tinyann/text_io.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <thread>
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

// Scalar reference kernels (for SIMD correctness checks).
float scalar_ip(const float* a, const float* b, std::size_t dim) {
    float s = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

float scalar_l2(const float* a, const float* b, std::size_t dim) {
    float s = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

float scalar_cosine(const float* a, const float* b, std::size_t dim) {
    float dot = 0.f, na = 0.f, nb = 0.f;
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

void test_simd_matches_scalar() {
    std::cout << "distance_backend=" << tinyann::distance_backend() << "\n";
    // Exercise multiple dimensions including non-multiples of 4/8 (tail loops).
    const std::size_t dims[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 256};
    std::mt19937_64 rng(2026);
    std::normal_distribution<float> nd(0.f, 1.f);

    for (std::size_t dim : dims) {
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = nd(rng);
            b[i] = nd(rng);
        }
        CHECK_NEAR(tinyann::inner_product(a.data(), b.data(), dim), scalar_ip(a.data(), b.data(), dim),
                   1e-4 * static_cast<double>(dim));
        CHECK_NEAR(tinyann::euclidean_distance(a.data(), b.data(), dim),
                   scalar_l2(a.data(), b.data(), dim), 1e-4 * std::sqrt(static_cast<double>(dim)));
        CHECK_NEAR(tinyann::cosine_similarity(a.data(), b.data(), dim),
                   scalar_cosine(a.data(), b.data(), dim), 1e-4);

        // Zero vector cosine
        std::vector<float> z(dim, 0.f);
        CHECK_NEAR(tinyann::cosine_similarity(z.data(), a.data(), dim), 0.0, 1e-6);
        CHECK_NEAR(tinyann::cosine_similarity(a.data(), z.data(), dim), 0.0, 1e-6);
        CHECK_NEAR(tinyann::euclidean_distance(a.data(), a.data(), dim), 0.0, 1e-5);
    }
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

/// Byte-identical SearchResult lists (id + float bit pattern).
bool results_byte_identical(const std::vector<tinyann::SearchResult>& a,
                            const std::vector<tinyann::SearchResult>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id) {
            return false;
        }
        std::uint32_t ua = 0, ub = 0;
        std::memcpy(&ua, &a[i].score, sizeof(float));
        std::memcpy(&ub, &b[i].score, sizeof(float));
        if (ua != ub) {
            return false;
        }
    }
    return true;
}

std::string temp_path(const char* name) {
#if defined(_WIN32)
    return std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") + "\\" + name;
#else
    return std::string("/tmp/") + name;
#endif
}

void test_exact_save_load_identical_search() {
    tinyann::Index idx(3, tinyann::Metric::Cosine);
    idx.add(10, {1.f, 0.f, 0.f});
    idx.add(20, {0.f, 1.f, 0.f});
    idx.add(30, {0.7f, 0.7f, 0.f});
    idx.add(40, {-1.f, 0.f, 0.f});

    const std::vector<std::vector<float>> queries = {
        {1.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {0.5f, 0.5f, 0.f},
        {0.f, 0.f, 1.f},
    };

    std::vector<std::vector<tinyann::SearchResult>> before;
    for (const auto& q : queries) {
        before.push_back(idx.search(q, 3));
    }

    const std::string path = temp_path("tinyann_exact_test.bin");
    idx.save(path);
    tinyann::Index loaded = tinyann::Index::load(path);
    CHECK(loaded.dimension() == idx.dimension());
    CHECK(loaded.metric() == idx.metric());
    CHECK(loaded.size() == idx.size());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto after = loaded.search(queries[i], 3);
        CHECK(results_byte_identical(before[i], after));
    }

    // Empty index round-trip
    tinyann::Index empty(2, tinyann::Metric::Euclidean);
    const std::string path_empty = temp_path("tinyann_exact_empty.bin");
    empty.save(path_empty);
    auto loaded_empty = tinyann::Index::load(path_empty);
    CHECK(loaded_empty.empty());
    CHECK(loaded_empty.search({1.f, 0.f}, 5).empty());
}

void test_exact_load_rejects_hnsw_file() {
    tinyann::HnswParams p;
    p.M = 4;
    p.ef_construction = 16;
    p.ef_search = 8;
    tinyann::HnswIndex h(2, tinyann::Metric::InnerProduct, p);
    h.add(1, {1.f, 0.f});
    const std::string path = temp_path("tinyann_hnsw_as_exact.bin");
    h.save(path);
    bool threw = false;
    try {
        (void)tinyann::Index::load(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_hnsw_save_load_identical_search() {
    tinyann::HnswParams p;
    p.M = 12;
    p.ef_construction = 100;
    p.ef_search = 40;
    p.seed = 77;

    const std::size_t dim = 16;
    const std::size_t n = 500;
    const std::size_t nq = 30;
    const std::size_t k = 10;

    tinyann::HnswIndex hnsw(dim, tinyann::Metric::Cosine, p);
    std::mt19937_64 rng(4242);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(static_cast<std::int64_t>(i), random_unit_vector(dim, rng));
    }

    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }

    std::vector<std::vector<tinyann::SearchResult>> before;
    for (const auto& q : queries) {
        before.push_back(hnsw.search(q, k, /*ef=*/40));
    }

    const std::string path = temp_path("tinyann_hnsw_test.bin");
    hnsw.save(path);
    tinyann::HnswIndex loaded = tinyann::HnswIndex::load(path);

    CHECK(loaded.dimension() == hnsw.dimension());
    CHECK(loaded.metric() == hnsw.metric());
    CHECK(loaded.size() == hnsw.size());
    CHECK(loaded.params().M == hnsw.params().M);
    CHECK(loaded.params().ef_construction == hnsw.params().ef_construction);
    CHECK(loaded.params().ef_search == hnsw.params().ef_search);

    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto after = loaded.search(queries[i], k, /*ef=*/40);
        CHECK(results_byte_identical(before[i], after));
    }

    // Default search() also identical
    for (std::size_t i = 0; i < queries.size(); ++i) {
        CHECK(results_byte_identical(hnsw.search(queries[i], k), loaded.search(queries[i], k)));
    }
}

void test_hnsw_load_rejects_exact_file() {
    tinyann::Index idx(2, tinyann::Metric::Cosine);
    idx.add(1, {1.f, 0.f});
    const std::string path = temp_path("tinyann_exact_as_hnsw.bin");
    idx.save(path);
    bool threw = false;
    try {
        (void)tinyann::HnswIndex::load(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

/// Read whole file into a buffer (for corrupting HNSW payloads in tests).
std::vector<char> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
}

void write_file_bytes(const std::string& path, const std::vector<char>& buf) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
}

/// Byte offset of entry_point (int32) in an HNSW save file (after params/seed).
/// Layout: header | metric,dim,count,ids,data | M,efc,efs,seed | entry_point | ...
std::size_t hnsw_entry_point_offset(const std::vector<char>& buf) {
    std::size_t off = 0;
    auto need = [&](std::size_t n) {
        if (off + n > buf.size()) {
            throw std::runtime_error("hnsw test helper: truncated buffer");
        }
    };
    need(4 + 4 + 4);  // magic, version, kind
    off += 12;
    need(4);
    off += 4;  // metric
    need(16);
    std::uint64_t dim = 0;
    std::uint64_t count = 0;
    std::memcpy(&dim, buf.data() + off, 8);
    off += 8;
    std::memcpy(&count, buf.data() + off, 8);
    off += 8;
    need(count * 8 + count * dim * 4 + 8 * 4);
    off += count * 8;          // ids
    off += count * dim * 4;    // floats
    off += 8 * 4;              // M, efc, efs, seed
    return off;
}

bool hnsw_load_throws(const std::string& path) {
    try {
        (void)tinyann::HnswIndex::load(path);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

void test_hnsw_load_rejects_corrupt_graph() {
    tinyann::HnswParams p;
    p.M = 4;
    p.ef_construction = 16;
    p.ef_search = 8;
    p.seed = 1;
    tinyann::HnswIndex h(2, tinyann::Metric::InnerProduct, p);
    h.add(1, {1.f, 0.f});
    h.add(2, {0.f, 1.f});
    h.add(3, {0.5f, 0.f});

    const std::string good = temp_path("tinyann_hnsw_graph_good.bin");
    h.save(good);
    // Sanity: good file still loads and searches.
    {
        auto loaded = tinyann::HnswIndex::load(good);
        CHECK(loaded.size() == 3);
        auto hits = loaded.search({1.f, 0.f}, 1);
        CHECK(hits.size() == 1);
        CHECK(hits[0].id == 1);
    }

    const auto base = read_file_bytes(good);
    const std::size_t ep_off = hnsw_entry_point_offset(base);

    // 1) entry_point out of range
    {
        auto buf = base;
        const std::int32_t bad_ep = 999;
        std::memcpy(buf.data() + ep_off, &bad_ep, sizeof(bad_ep));
        const std::string path = temp_path("tinyann_hnsw_bad_ep.bin");
        write_file_bytes(path, buf);
        CHECK(hnsw_load_throws(path));
    }

    // 2) negative entry_point (non-empty)
    {
        auto buf = base;
        const std::int32_t bad_ep = -1;
        std::memcpy(buf.data() + ep_off, &bad_ep, sizeof(bad_ep));
        const std::string path = temp_path("tinyann_hnsw_neg_ep.bin");
        write_file_bytes(path, buf);
        CHECK(hnsw_load_throws(path));
    }

    // 3) max_level inconsistent with node levels (entry still at real max)
    {
        auto buf = base;
        std::int32_t max_level = 0;
        std::memcpy(&max_level, buf.data() + ep_off + 4, sizeof(max_level));
        const std::int32_t bad_ml = max_level + 50;
        std::memcpy(buf.data() + ep_off + 4, &bad_ml, sizeof(bad_ml));
        const std::string path = temp_path("tinyann_hnsw_bad_maxlevel.bin");
        write_file_bytes(path, buf);
        CHECK(hnsw_load_throws(path));
    }

    // 4) OOB neighbor index: patch first neighbor int32 after first node's layer header.
    // After entry_point(4)+max_level(4)+levels(n*4), first node writes n_layers u32, then
    // for layer 0: n_links u32, then n_links * i32 neighbors.
    {
        auto buf = base;
        const std::size_t n = 3;
        std::size_t off = ep_off + 8;  // past entry + max_level
        off += n * 4;                  // levels
        // n_layers for node 0
        std::uint32_t n_layers = 0;
        std::memcpy(&n_layers, buf.data() + off, 4);
        off += 4;
        CHECK(n_layers >= 1);
        std::uint32_t n_links = 0;
        std::memcpy(&n_links, buf.data() + off, 4);
        off += 4;
        if (n_links > 0) {
            const std::int32_t bad_nb = 999;
            std::memcpy(buf.data() + off, &bad_nb, sizeof(bad_nb));
            const std::string path = temp_path("tinyann_hnsw_bad_neighbor.bin");
            write_file_bytes(path, buf);
            CHECK(hnsw_load_throws(path));
        } else {
            // Degenerate: force a self-loop style corruption on levels layer count instead.
            // neighbors_[0].size() mismatch: set n_layers to 0 while level may be >= 0.
            off = ep_off + 8 + n * 4;
            const std::uint32_t zero_layers = 0;
            std::memcpy(buf.data() + off, &zero_layers, 4);
            // File may now be misaligned for remaining nodes; load should still fail validation
            // or fail mid-read. Either is acceptable rejection.
            const std::string path = temp_path("tinyann_hnsw_bad_layers.bin");
            write_file_bytes(path, buf);
            CHECK(hnsw_load_throws(path));
        }
    }

    // 5) layer count mismatch: set node 0's n_layers to level+1+1 without enough payload —
    //    simpler: rewrite levels[0] to a huge level so neighbors size won't match after read.
    //    Changing only levels[0] to levels[0]+1 makes neighbors_[0].size() != level+1.
    {
        auto buf = base;
        const std::size_t levels_off = ep_off + 8;
        std::int32_t lvl0 = 0;
        std::memcpy(&lvl0, buf.data() + levels_off, 4);
        const std::int32_t bad_lvl = lvl0 + 1;
        std::memcpy(buf.data() + levels_off, &bad_lvl, 4);
        // max_level / entry may also become inconsistent; either way load must reject.
        const std::string path = temp_path("tinyann_hnsw_bad_level_field.bin");
        write_file_bytes(path, buf);
        CHECK(hnsw_load_throws(path));
    }

    // 6) Hostile adjacency length: n_links > n must be rejected *before* huge allocate.
    {
        auto buf = base;
        const std::size_t n_nodes = 3;
        std::size_t off = ep_off + 8 + n_nodes * 4;  // past entry, max_level, levels
        std::uint32_t n_layers = 0;
        std::memcpy(&n_layers, buf.data() + off, 4);
        off += 4;
        CHECK(n_layers >= 1);
        // Overwrite first layer's n_links with a value larger than n (but small enough for test
        // file — rejection is on the count, before resize of an absurd allocation).
        const std::uint32_t huge_links = 100;  // > n=3
        std::memcpy(buf.data() + off, &huge_links, 4);
        const std::string path = temp_path("tinyann_hnsw_huge_links.bin");
        write_file_bytes(path, buf);
        CHECK(hnsw_load_throws(path));
    }

    // 7) Absurd node level (above kMaxHnswLevel=64) rejected early.
    {
        auto buf = base;
        const std::size_t levels_off = ep_off + 8;
        const std::int32_t absurd = 10'000;
        std::memcpy(buf.data() + levels_off, &absurd, 4);
        const std::string path = temp_path("tinyann_hnsw_absurd_level.bin");
        write_file_bytes(path, buf);
        CHECK(hnsw_load_throws(path));
    }
}

void test_persistence_all_metrics() {
    for (auto metric : {tinyann::Metric::Cosine, tinyann::Metric::Euclidean,
                        tinyann::Metric::InnerProduct}) {
        tinyann::Index exact(2, metric);
        exact.add(1, {1.f, 0.f});
        exact.add(2, {0.f, 1.f});
        exact.add(3, {0.5f, 0.5f});
        const auto q = std::vector<float>{1.f, 0.f};
        const auto before = exact.search(q, 2);
        const std::string path = temp_path("tinyann_metric_roundtrip.bin");
        exact.save(path);
        const auto after = tinyann::Index::load(path).search(q, 2);
        CHECK(results_byte_identical(before, after));

        tinyann::HnswParams p;
        p.M = 4;
        p.ef_construction = 32;
        p.ef_search = 16;
        p.seed = 3;
        tinyann::HnswIndex h(2, metric, p);
        h.add(1, {1.f, 0.f});
        h.add(2, {0.f, 1.f});
        h.add(3, {0.5f, 0.5f});
        const auto hb = h.search(q, 2);
        h.save(path);
        const auto ha = tinyann::HnswIndex::load(path).search(q, 2);
        CHECK(results_byte_identical(hb, ha));
    }
}

bool results_contain_id(const std::vector<tinyann::SearchResult>& r, std::int64_t id) {
    for (const auto& hit : r) {
        if (hit.id == id) {
            return true;
        }
    }
    return false;
}

void test_exact_remove_update() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    idx.add(1, {1.f, 0.f});
    idx.add(2, {0.f, 1.f});
    idx.add(3, {0.5f, 0.f});
    CHECK(idx.size() == 3);
    CHECK(idx.contains(2));
    CHECK(!idx.remove(99));
    CHECK(idx.remove(2));
    CHECK(!idx.contains(2));
    CHECK(idx.size() == 2);
    auto r = idx.search({0.f, 1.f}, 10);
    CHECK(!results_contain_id(r, 2));
    CHECK(results_contain_id(r, 1) || results_contain_id(r, 3));

    // Update remaining id=1
    CHECK(idx.update(1, {0.f, 2.f}));
    CHECK(!idx.update(2, {1.f, 0.f}));  // gone
    auto r2 = idx.search({0.f, 1.f}, 1);
    CHECK(r2.size() == 1);
    CHECK(r2[0].id == 1);
    CHECK_NEAR(r2[0].score, 2.0, 1e-5);

    // Remove all
    CHECK(idx.remove(1));
    CHECK(idx.remove(3));
    CHECK(idx.empty());
    CHECK(idx.search({1.f, 0.f}, 5).empty());

    // Duplicate ids: remove clears all
    idx.add(7, {1.f, 0.f});
    idx.add(7, {2.f, 0.f});
    CHECK(idx.size() == 2);
    CHECK(idx.remove(7));
    CHECK(idx.empty());
}

void test_hnsw_remove_update() {
    tinyann::HnswParams p;
    p.M = 8;
    p.ef_construction = 64;
    p.ef_search = 32;
    p.seed = 5;
    tinyann::HnswIndex h(2, tinyann::Metric::InnerProduct, p);
    h.add(1, {1.f, 0.f});
    h.add(2, {0.f, 1.f});
    h.add(3, {0.5f, 0.f});
    h.add(4, {-1.f, 0.f});
    CHECK(h.size() == 4);
    CHECK(h.contains(3));
    CHECK(!h.remove(99));
    CHECK(h.remove(3));
    CHECK(!h.contains(3));
    CHECK(h.size() == 3);

    auto r = h.search({1.f, 0.f}, 10);
    CHECK(r.size() == 3);
    CHECK(!results_contain_id(r, 3));

    CHECK(h.update(1, {0.f, 3.f}));
    CHECK(!h.update(3, {1.f, 0.f}));
    auto r2 = h.search({0.f, 1.f}, 1);
    CHECK(r2.size() == 1);
    CHECK(r2[0].id == 1);
    CHECK_NEAR(r2[0].score, 3.0, 1e-5);
}

void test_hnsw_remove_entry_point() {
    // Deterministic seed; delete every id and ensure we never crash and
    // entry/search stay valid when non-empty.
    tinyann::HnswParams p;
    p.M = 6;
    p.ef_construction = 50;
    p.ef_search = 20;
    p.seed = 11;
    tinyann::HnswIndex h(4, tinyann::Metric::Cosine, p);
    std::mt19937_64 rng(88);
    const std::size_t n = 40;
    for (std::size_t i = 0; i < n; ++i) {
        h.add(static_cast<std::int64_t>(i), random_unit_vector(4, rng));
    }
    // Delete in an order that will hit the entry point eventually / repeatedly.
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t id = static_cast<std::int64_t>((i * 7) % n);
        if (!h.contains(id)) {
            continue;
        }
        CHECK(h.remove(id));
        CHECK(!h.contains(id));
        if (!h.empty()) {
            auto hits = h.search(random_unit_vector(4, rng), std::min<std::size_t>(5, h.size()));
            CHECK(!hits.empty());
            CHECK(hits.size() <= h.size());
            for (const auto& hit : hits) {
                CHECK(hit.id != id);
                CHECK(h.contains(hit.id));
            }
        } else {
            CHECK(h.search({1.f, 0.f, 0.f, 0.f}, 3).empty());
        }
    }
    CHECK(h.empty());
}

void test_hnsw_remove_keeps_searchable() {
    tinyann::HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    p.ef_search = 50;
    p.seed = 123;
    const std::size_t dim = 16;
    const std::size_t n = 300;
    tinyann::HnswIndex h(dim, tinyann::Metric::Euclidean, p);
    tinyann::Index exact(dim, tinyann::Metric::Euclidean);
    std::mt19937_64 rng(55);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = random_unit_vector(dim, rng);
        // Scale for L2 variety
        for (float& x : v) {
            x *= 2.f;
        }
        h.add(static_cast<std::int64_t>(i), v);
        exact.add(static_cast<std::int64_t>(i), v);
    }

    // Remove a chunk of ids including low and high indices
    for (std::int64_t id = 0; id < 50; ++id) {
        CHECK(h.remove(id));
        CHECK(exact.remove(id));
    }
    for (std::int64_t id = 250; id < 300; ++id) {
        CHECK(h.remove(id));
        CHECK(exact.remove(id));
    }
    CHECK(h.size() == 200);
    CHECK(exact.size() == 200);

    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 20; ++i) {
        auto q = random_unit_vector(dim, rng);
        for (float& x : q) {
            x *= 2.f;
        }
        queries.push_back(q);
    }

    // No removed id appears
    for (const auto& q : queries) {
        auto hits = h.search(q, 10);
        CHECK(hits.size() == 10);
        for (const auto& hit : hits) {
            CHECK(hit.id >= 50);
            CHECK(hit.id < 250);
            CHECK(h.contains(hit.id));
        }
    }

    // Still reasonable recall after deletions
    const double rec = h.recall_at_k_vs(exact, queries, 10, 50);
    std::cout << "recall@10[after_remove]=" << rec << "\n";
    CHECK(rec > 0.85);
}

void test_exact_filtered_search() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    idx.add(1, {10.f, 0.f});
    idx.add(2, {9.f, 0.f});
    idx.add(3, {8.f, 0.f});
    idx.add(4, {7.f, 0.f});
    const auto q = std::vector<float>{1.f, 0.f};

    // Pass everything == unfiltered
    auto all_pass = idx.search(q, 3, [](std::int64_t) { return true; });
    auto unf = idx.search(q, 3);
    CHECK(results_byte_identical(all_pass, unf));
    CHECK(all_pass[0].id == 1);

    // Only even ids
    auto even = idx.search(q, 10, [](std::int64_t id) { return id % 2 == 0; });
    CHECK(even.size() == 2);
    CHECK(even[0].id == 2);
    CHECK(even[1].id == 4);

    // k larger than eligible
    auto even_k1 = idx.search(q, 1, [](std::int64_t id) { return id % 2 == 0; });
    CHECK(even_k1.size() == 1);
    CHECK(even_k1[0].id == 2);

    // Reject all -> empty
    auto none = idx.search(q, 5, [](std::int64_t) { return false; });
    CHECK(none.empty());

    // k == 0
    CHECK(idx.search(q, 0, [](std::int64_t) { return true; }).empty());
}

void test_hnsw_filtered_search_basic() {
    tinyann::HnswParams p;
    p.M = 8;
    p.ef_construction = 64;
    p.ef_search = 32;
    p.seed = 2;
    tinyann::HnswIndex h(2, tinyann::Metric::InnerProduct, p);
    h.add(1, {10.f, 0.f});
    h.add(2, {9.f, 0.f});
    h.add(3, {8.f, 0.f});
    h.add(4, {7.f, 0.f});
    const auto q = std::vector<float>{1.f, 0.f};

    auto all_pass = h.search(q, 3, [](std::int64_t) { return true; });
    auto unf = h.search(q, 3);
    CHECK(results_byte_identical(all_pass, unf));

    auto even = h.search(q, 10, [](std::int64_t id) { return (id % 2) == 0; });
    CHECK(even.size() == 2);
    for (const auto& hit : even) {
        CHECK((hit.id % 2) == 0);
    }
    CHECK(even[0].id == 2);
    CHECK(even[1].id == 4);

    CHECK(h.search(q, 5, [](std::int64_t) { return false; }).empty());
}

void test_hnsw_filtered_not_postfilter_topk() {
    // Construct a case where the true top-k under the filter are NOT the
    // unfiltered top-k. Post-filtering unfiltered top-1 would miss the only
    // eligible neighbor; in-graph filtering must still find it.
    //
    // ids: 1 closest overall but ineligible; 2 slightly farther but eligible.
    tinyann::HnswParams p;
    p.M = 4;
    p.ef_construction = 32;
    p.ef_search = 16;
    p.seed = 1;
    tinyann::HnswIndex h(2, tinyann::Metric::InnerProduct, p);
    h.add(1, {100.f, 0.f});  // best unfiltered
    h.add(2, {50.f, 0.f});   // best eligible
    h.add(3, {10.f, 0.f});
    const auto q = std::vector<float>{1.f, 0.f};

    auto unf = h.search(q, 1);
    CHECK(unf.size() == 1);
    CHECK(unf[0].id == 1);
    // Naive post-filter of top-1 would yield empty; proper filter returns id 2.
    auto filt = h.search(q, 1, [](std::int64_t id) { return id != 1; });
    CHECK(filt.size() == 1);
    CHECK(filt[0].id == 2);
    CHECK_NEAR(filt[0].score, 50.0, 1e-4);
}

void test_ivf_basic_and_filter() {
    tinyann::IvfParams p;
    p.nlist = 4;
    p.nprobe = 4;  // probe all ⇒ exact on tiny set
    p.kmeans_iters = 10;
    p.seed = 1;
    tinyann::IvfIndex ivf(2, tinyann::Metric::InnerProduct, p);
    std::vector<std::vector<float>> train = {{1.f, 0.f}, {0.f, 1.f}, {0.5f, 0.f}, {-1.f, 0.f}};
    ivf.train(train);
    CHECK(ivf.trained());
    for (std::size_t i = 0; i < train.size(); ++i) {
        ivf.add(static_cast<std::int64_t>(i + 1), train[i]);
    }
    auto r = ivf.search({1.f, 0.f}, 2);
    CHECK(r.size() == 2);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 1.0, 1e-5);

    auto none = ivf.search({1.f, 0.f}, 5, [](std::int64_t) { return false; });
    CHECK(none.empty());
    auto even = ivf.search({1.f, 0.f}, 10, [](std::int64_t id) { return id % 2 == 0; });
    for (const auto& h : even) {
        CHECK((h.id % 2) == 0);
    }
    CHECK(!even.empty());
}

void test_ivf_recall() {
    const std::size_t dim = 32;
    const std::size_t n = 2000;
    const std::size_t nq = 40;
    const std::size_t k = 10;
    tinyann::IvfParams p;
    p.nlist = 50;
    p.nprobe = 20;  // 40% of lists
    p.kmeans_iters = 20;
    p.seed = 7;

    for (auto metric : {tinyann::Metric::Cosine, tinyann::Metric::Euclidean,
                        tinyann::Metric::InnerProduct}) {
        tinyann::Index exact(dim, metric);
        tinyann::IvfIndex ivf(dim, metric, p);
        std::mt19937_64 rng(100 + static_cast<unsigned>(metric));
        std::vector<std::vector<float>> all;
        all.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            all.push_back(random_unit_vector(dim, rng));
        }
        ivf.train(all);
        for (std::size_t i = 0; i < n; ++i) {
            exact.add(static_cast<std::int64_t>(i), all[i]);
            ivf.add(static_cast<std::int64_t>(i), all[i]);
        }
        std::vector<std::vector<float>> queries;
        for (std::size_t i = 0; i < nq; ++i) {
            queries.push_back(random_unit_vector(dim, rng));
        }
        const double rec = ivf.recall_at_k_vs(exact, queries, k);
        std::cout << "recall@10[ivf_" << tinyann::metric_name(metric) << "]=" << rec << "\n";
        CHECK(rec > 0.85);
    }
}

void test_ivf_save_load() {
    tinyann::IvfParams p;
    p.nlist = 8;
    p.nprobe = 4;
    p.seed = 3;
    tinyann::IvfIndex ivf(4, tinyann::Metric::Cosine, p);
    std::mt19937_64 rng(11);
    std::vector<std::vector<float>> all;
    for (int i = 0; i < 80; ++i) {
        all.push_back(random_unit_vector(4, rng));
    }
    ivf.train(all);
    for (int i = 0; i < 80; ++i) {
        ivf.add(i, all[static_cast<std::size_t>(i)]);
    }
    const auto q = random_unit_vector(4, rng);
    const auto before = ivf.search(q, 5);
    const std::string path = temp_path("tinyann_ivf.bin");
    ivf.save(path);
    auto loaded = tinyann::IvfIndex::load(path);
    CHECK(loaded.trained());
    CHECK(loaded.size() == ivf.size());
    CHECK(results_byte_identical(before, loaded.search(q, 5)));
}

/// Byte offset of first list_of_ u32 in an IVF save file.
/// Layout: header | metric,dim,count,ids,data | nlist,nprobe,iters,seed | centroids | list_of...
std::size_t ivf_list_of_offset(const std::vector<char>& buf) {
    std::size_t off = 0;
    auto need = [&](std::size_t n) {
        if (off + n > buf.size()) {
            throw std::runtime_error("ivf test helper: truncated buffer");
        }
    };
    need(12);
    off += 12;  // magic, version, kind
    need(4);
    off += 4;  // metric
    need(16);
    std::uint64_t dim = 0;
    std::uint64_t count = 0;
    std::memcpy(&dim, buf.data() + off, 8);
    off += 8;
    std::memcpy(&count, buf.data() + off, 8);
    off += 8;
    need(count * 8 + count * dim * 4 + 8 * 4);
    off += count * 8;
    off += count * dim * 4;
    std::uint64_t nlist = 0;
    std::memcpy(&nlist, buf.data() + off, 8);
    off += 8;       // nlist
    off += 8 * 3;   // nprobe, kmeans_iters, seed
    need(nlist * dim * 4);
    off += nlist * dim * 4;  // centroids
    return off;
}

bool ivf_load_throws(const std::string& path) {
    try {
        (void)tinyann::IvfIndex::load(path);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

/// Craft a minimal exact-index header with hostile count/dimension (no vector payload).
void write_hostile_exact_header(const std::string& path, std::uint64_t dim, std::uint64_t count) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const char magic[4] = {'T', 'A', 'N', 'N'};
    out.write(magic, 4);
    const std::uint32_t ver = 1;
    const std::uint32_t kind = 1;  // exact
    const std::uint32_t metric = 0;
    out.write(reinterpret_cast<const char*>(&ver), 4);
    out.write(reinterpret_cast<const char*>(&kind), 4);
    out.write(reinterpret_cast<const char*>(&metric), 4);
    out.write(reinterpret_cast<const char*>(&dim), 8);
    out.write(reinterpret_cast<const char*>(&count), 8);
}

void write_hostile_sq_header(const std::string& path, std::uint64_t dim, std::uint64_t count) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const char magic[4] = {'T', 'A', 'N', 'N'};
    out.write(magic, 4);
    const std::uint32_t ver = 1;
    const std::uint32_t kind = 4;  // SQ
    const std::uint32_t metric = 0;
    out.write(reinterpret_cast<const char*>(&ver), 4);
    out.write(reinterpret_cast<const char*>(&kind), 4);
    out.write(reinterpret_cast<const char*>(&metric), 4);
    out.write(reinterpret_cast<const char*>(&dim), 8);
    out.write(reinterpret_cast<const char*>(&count), 8);
}

void test_hnsw_concurrent_search() {
    // Concurrent search on one HnswIndex must match sequential results (no races on
    // visit stamps / query norm). Writers are not concurrent here — build then search-only.
    const std::size_t dim = 32;
    const std::size_t n = 3000;
    const std::size_t nq = 80;
    const std::size_t k = 10;
    const std::size_t nthreads = 8;
    const std::size_t rounds = 4;

    tinyann::HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    p.ef_search = 64;
    p.seed = 2026;

    tinyann::HnswIndex hnsw(dim, tinyann::Metric::Cosine, p);
    std::mt19937_64 rng(4242);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(static_cast<std::int64_t>(i), random_unit_vector(dim, rng));
    }

    std::vector<std::vector<float>> queries;
    queries.reserve(nq);
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }

    // Sequential gold standard (same queries, same k/ef).
    std::vector<std::vector<tinyann::SearchResult>> gold(nq);
    for (std::size_t i = 0; i < nq; ++i) {
        gold[i] = hnsw.search(queries[i], k, /*ef=*/64);
        CHECK(!gold[i].empty());
    }

    auto even = [](std::int64_t id) { return (id % 2) == 0; };
    std::vector<std::vector<tinyann::SearchResult>> filtered_gold(nq);
    for (std::size_t i = 0; i < nq; ++i) {
        filtered_gold[i] = hnsw.search(queries[i], k, /*ef=*/64, even);
        for (const auto& hit : filtered_gold[i]) {
            CHECK((hit.id % 2) == 0);
        }
    }

    std::atomic<int> mismatches{0};
    std::atomic<int> filtered_mismatches{0};

    auto worker = [&](std::size_t tid) {
        for (std::size_t r = 0; r < rounds; ++r) {
            for (std::size_t i = 0; i < nq; ++i) {
                // Stagger query order slightly per thread to stress interleaving.
                const std::size_t qi = (i * 17 + tid * 3 + r) % nq;
                const auto hits = hnsw.search(queries[qi], k, /*ef=*/64);
                if (!results_byte_identical(hits, gold[qi])) {
                    mismatches.fetch_add(1, std::memory_order_relaxed);
                }
                const auto fh = hnsw.search(queries[qi], k, /*ef=*/64, even);
                if (!results_byte_identical(fh, filtered_gold[qi])) {
                    filtered_mismatches.fetch_add(1, std::memory_order_relaxed);
                }
                for (const auto& hit : fh) {
                    if ((hit.id % 2) != 0) {
                        filtered_mismatches.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (std::size_t t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
        th.join();
    }

    CHECK(mismatches.load() == 0);
    CHECK(filtered_mismatches.load() == 0);

    // After concurrent searches, single-thread results still match gold (no corruption).
    for (std::size_t i = 0; i < nq; ++i) {
        CHECK(results_byte_identical(hnsw.search(queries[i], k, /*ef=*/64), gold[i]));
        CHECK(results_byte_identical(hnsw.search(queries[i], k, /*ef=*/64, even), filtered_gold[i]));
    }

    std::cout << "hnsw_concurrent_search threads=" << nthreads << " nq=" << nq
              << " rounds=" << rounds << " OK\n";
}

void test_reject_non_finite() {
    tinyann::Index idx(2, tinyann::Metric::InnerProduct);
    bool threw = false;
    try {
        idx.add(1, {1.f, std::numeric_limits<float>::quiet_NaN()});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(idx.empty());

    idx.add(1, {1.f, 0.f});
    threw = false;
    try {
        (void)idx.search({std::numeric_limits<float>::infinity(), 0.f}, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    tinyann::HnswParams p;
    p.M = 4;
    p.ef_construction = 8;
    p.ef_search = 4;
    tinyann::HnswIndex h(2, tinyann::Metric::Cosine, p);
    threw = false;
    try {
        h.add(1, {0.f, std::numeric_limits<float>::quiet_NaN()});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_parse_vector_text_line_int64_ids() {
    bool has_id = false;
    std::int64_t id = 0;
    std::vector<float> vec;

    // Large id must not go through float (2^24+1).
    CHECK(tinyann::parse_vector_text_line("16777217 1.0 0.0", 2, true, has_id, id, vec));
    CHECK(has_id);
    CHECK(id == 16777217);
    CHECK(vec.size() == 2);
    CHECK_NEAR(vec[0], 1.0, 1e-6);
    CHECK_NEAR(vec[1], 0.0, 1e-6);

    // Negative ids are first-class (not treated as "missing id").
    CHECK(tinyann::parse_vector_text_line("-5 3.0 4.0", 2, true, has_id, id, vec));
    CHECK(has_id);
    CHECK(id == -5);
    CHECK_NEAR(vec[0], 3.0, 1e-6);
    CHECK_NEAR(vec[1], 4.0, 1e-6);

    // No id column
    CHECK(tinyann::parse_vector_text_line("1.5 -2.5", 2, true, has_id, id, vec));
    CHECK(!has_id);
    CHECK(vec.size() == 2);

    // Reject non-integer id tokens (old float-style ids)
    CHECK(!tinyann::parse_vector_text_line("1.0 1.0 0.0", 2, true, has_id, id, vec));
    CHECK(!tinyann::parse_vector_text_line("1e2 1.0 0.0", 2, true, has_id, id, vec));

    // Wrong arity
    CHECK(!tinyann::parse_vector_text_line("1 2 3 4", 2, true, has_id, id, vec));
    CHECK(!tinyann::parse_vector_text_line("", 2, true, has_id, id, vec));

    // Very large int64 still exact as text
    CHECK(tinyann::parse_vector_text_line("9223372036854775806 0.0", 1, true, has_id, id, vec));
    CHECK(has_id);
    CHECK(id == 9223372036854775806LL);
}

void test_load_rejects_size_overflow() {
    // count * dimension overflows size_t on 64-bit (and would wrap to a small value without checks).
    const std::uint64_t huge = (std::uint64_t{1} << 32);
    {
        const std::string path = temp_path("tinyann_overflow_exact.bin");
        write_hostile_exact_header(path, huge, huge);
        bool threw = false;
        try {
            (void)tinyann::Index::load(path);
        } catch (const std::runtime_error& e) {
            threw = true;
            const std::string msg = e.what();
            CHECK(msg.find("overflow") != std::string::npos ||
                  msg.find("size") != std::string::npos);
        }
        CHECK(threw);
    }
    {
        const std::string path = temp_path("tinyann_overflow_sq.bin");
        write_hostile_sq_header(path, huge, huge);
        bool threw = false;
        try {
            (void)tinyann::IndexSq::load(path);
        } catch (const std::runtime_error& e) {
            threw = true;
            const std::string msg = e.what();
            CHECK(msg.find("overflow") != std::string::npos ||
                  msg.find("size") != std::string::npos);
        }
        CHECK(threw);
    }

    // IVF: huge nlist * dimension for centroids (after a tiny valid vector block).
    // Build a minimal IVF-shaped file: header + empty vectors + huge nlist.
    {
        const std::string path = temp_path("tinyann_overflow_ivf_cent.bin");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const char magic[4] = {'T', 'A', 'N', 'N'};
        out.write(magic, 4);
        const std::uint32_t ver = 1;
        const std::uint32_t kind = 3;  // IVF
        const std::uint32_t metric = 0;
        const std::uint64_t dim = 64;
        const std::uint64_t count = 0;
        out.write(reinterpret_cast<const char*>(&ver), 4);
        out.write(reinterpret_cast<const char*>(&kind), 4);
        out.write(reinterpret_cast<const char*>(&metric), 4);
        out.write(reinterpret_cast<const char*>(&dim), 8);
        out.write(reinterpret_cast<const char*>(&count), 8);
        // nlist * dim must overflow size_t (dim=64 above).
        const std::uint64_t nlist =
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 64) + 1;
        const std::uint64_t nprobe = 1;
        const std::uint64_t iters = 1;
        const std::uint64_t seed = 1;
        out.write(reinterpret_cast<const char*>(&nlist), 8);
        out.write(reinterpret_cast<const char*>(&nprobe), 8);
        out.write(reinterpret_cast<const char*>(&iters), 8);
        out.write(reinterpret_cast<const char*>(&seed), 8);
        out.close();

        bool threw = false;
        try {
            (void)tinyann::IvfIndex::load(path);
        } catch (const std::runtime_error&) {
            threw = true;
        } catch (const std::invalid_argument&) {
            // ctor may reject absurd params in future; either way load must not succeed.
            threw = true;
        }
        CHECK(threw);
    }

    // Normal small exact file still loads (regression).
    {
        tinyann::Index idx(2, tinyann::Metric::Cosine);
        idx.add(1, {1.f, 0.f});
        const std::string path = temp_path("tinyann_overflow_control.bin");
        idx.save(path);
        auto loaded = tinyann::Index::load(path);
        CHECK(loaded.size() == 1);
        CHECK(loaded.search({1.f, 0.f}, 1)[0].id == 1);
    }
}

void test_ivf_load_rejects_corrupt_lists() {
    tinyann::IvfParams p;
    p.nlist = 4;
    p.nprobe = 4;
    p.kmeans_iters = 10;
    p.seed = 1;
    tinyann::IvfIndex ivf(2, tinyann::Metric::InnerProduct, p);
    std::vector<std::vector<float>> train = {{1.f, 0.f}, {0.f, 1.f}, {0.5f, 0.f}, {-1.f, 0.f}};
    ivf.train(train);
    for (std::size_t i = 0; i < train.size(); ++i) {
        ivf.add(static_cast<std::int64_t>(i + 1), train[i]);
    }

    const std::string good = temp_path("tinyann_ivf_lists_good.bin");
    ivf.save(good);
    {
        auto loaded = tinyann::IvfIndex::load(good);
        CHECK(loaded.size() == 4);
        auto hits = loaded.search({1.f, 0.f}, 1);
        CHECK(hits.size() == 1);
        CHECK(hits[0].id == 1);
    }

    const auto base = read_file_bytes(good);
    const std::size_t list_of_off = ivf_list_of_offset(base);
    const std::size_t n = 4;

    // 1) list_of_[0] out of range (>= nlist)
    {
        auto buf = base;
        const std::uint32_t bad = 999;
        std::memcpy(buf.data() + list_of_off, &bad, sizeof(bad));
        const std::string path = temp_path("tinyann_ivf_bad_list_of.bin");
        write_file_bytes(path, buf);
        CHECK(ivf_load_throws(path));
    }

    // 2) OOB node index in first inverted list (after list_of block)
    {
        auto buf = base;
        std::size_t off = list_of_off + n * 4;  // past list_of
        // lists_[0]: size u32, then nodes
        std::uint32_t sz0 = 0;
        std::memcpy(&sz0, buf.data() + off, 4);
        off += 4;
        if (sz0 > 0) {
            const std::int32_t bad_node = 999;
            std::memcpy(buf.data() + off, &bad_node, sizeof(bad_node));
            const std::string path = temp_path("tinyann_ivf_bad_list_node.bin");
            write_file_bytes(path, buf);
            CHECK(ivf_load_throws(path));
        } else {
            // Empty first list: corrupt by claiming size 1 and writing OOB (may need extra bytes).
            // Fall back: duplicate-style — set list_of[0] to a valid other list id without
            // updating lists (inconsistency between list_of and lists).
            auto buf2 = base;
            std::uint32_t orig = 0;
            std::memcpy(&orig, buf2.data() + list_of_off, 4);
            const std::uint32_t flipped = (orig + 1) % 4;
            std::memcpy(buf2.data() + list_of_off, &flipped, 4);
            const std::string path = temp_path("tinyann_ivf_list_of_lists_mismatch.bin");
            write_file_bytes(path, buf2);
            CHECK(ivf_load_throws(path));
        }
    }

    // 3) list_of disagrees with lists (valid range but wrong assignment)
    {
        auto buf = base;
        std::uint32_t orig = 0;
        std::memcpy(&orig, buf.data() + list_of_off, 4);
        const std::uint32_t flipped = (orig + 1) % static_cast<std::uint32_t>(4);
        if (flipped == orig) {
            // nlist edge: force 0 vs 1
        }
        std::memcpy(buf.data() + list_of_off, &flipped, 4);
        const std::string path = temp_path("tinyann_ivf_assignment_mismatch.bin");
        write_file_bytes(path, buf);
        CHECK(ivf_load_throws(path));
    }
}

void test_sq_quantize_dequantize() {
    const std::vector<float> v = {0.f, 1.f, -1.f, 0.5f, -0.25f, 127.f, -64.f};
    std::vector<std::int8_t> codes;
    float scale = 0.f;
    tinyann::sq::quantize(v, codes, scale);
    CHECK(codes.size() == v.size());
    CHECK(scale > 0.f);
    auto recon = tinyann::sq::dequantize(codes, scale);
    CHECK(recon.size() == v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        // Symmetric int8 should reconstruct within about one quantum.
        CHECK_NEAR(recon[i], v[i], scale * 0.51 + 1e-5);
    }
    // Zero vector
    std::vector<float> z(8, 0.f);
    float zs = 0.f;
    std::vector<std::int8_t> zc;
    tinyann::sq::quantize(z, zc, zs);
    CHECK_NEAR(zs, 1.0, 1e-6);
    for (auto c : zc) {
        CHECK(c == 0);
    }
}

void test_index_sq_search_and_recall() {
    tinyann::IndexSq sq(2, tinyann::Metric::InnerProduct);
    tinyann::Index exact(2, tinyann::Metric::InnerProduct);
    sq.add(1, {1.f, 0.f});
    sq.add(2, {0.f, 1.f});
    sq.add(3, {0.5f, 0.f});
    exact.add(1, {1.f, 0.f});
    exact.add(2, {0.f, 1.f});
    exact.add(3, {0.5f, 0.f});
    auto r = sq.search({1.f, 0.f}, 2);
    CHECK(r.size() == 2);
    CHECK(r[0].id == 1);

    // Larger random set: high recall vs float exact (quantization noise)
    const std::size_t dim = 32;
    const std::size_t n = 500;
    const std::size_t nq = 30;
    tinyann::IndexSq sq2(dim, tinyann::Metric::Cosine);
    tinyann::Index ex2(dim, tinyann::Metric::Cosine);
    std::mt19937_64 rng(55);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = random_unit_vector(dim, rng);
        sq2.add(static_cast<std::int64_t>(i), v);
        ex2.add(static_cast<std::int64_t>(i), v);
    }
    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }
    const double rec = sq2.recall_at_k_vs(ex2, queries, 10);
    std::cout << "recall@10[sq_int8]=" << rec << "\n";
    CHECK(rec > 0.95);

    CHECK(sq2.remove(0));
    CHECK(!sq2.contains(0));
}

void test_index_sq_save_load() {
    tinyann::IndexSq sq(3, tinyann::Metric::Euclidean);
    sq.add(10, {1.f, 2.f, 3.f});
    sq.add(20, {0.f, -1.f, 0.5f});
    const auto q = std::vector<float>{1.f, 0.f, 0.f};
    const auto before = sq.search(q, 2);
    const std::string path = temp_path("tinyann_sq.bin");
    sq.save(path);
    auto loaded = tinyann::IndexSq::load(path);
    CHECK(loaded.size() == 2);
    CHECK(results_byte_identical(before, loaded.search(q, 2)));
    // Reconstruct roughly matches original
    auto rec = loaded.reconstruct(0);
    CHECK_NEAR(rec[0], 1.f, 0.05);
}

void test_ivf_remove_update() {
    tinyann::IvfParams p;
    p.nlist = 4;
    p.nprobe = 4;
    p.seed = 2;
    tinyann::IvfIndex ivf(2, tinyann::Metric::InnerProduct, p);
    std::vector<std::vector<float>> t = {{1.f, 0.f}, {0.f, 1.f}, {0.5f, 0.5f}, {0.f, -1.f}};
    ivf.train(t);
    ivf.add(1, t[0]);
    ivf.add(2, t[1]);
    ivf.add(3, t[2]);
    CHECK(ivf.remove(2));
    CHECK(!ivf.contains(2));
    auto r = ivf.search({0.f, 1.f}, 10);
    CHECK(!results_contain_id(r, 2));
    CHECK(ivf.update(1, {0.f, 2.f}));
    auto r2 = ivf.search({0.f, 1.f}, 1);
    CHECK(r2[0].id == 1);
    CHECK_NEAR(r2[0].score, 2.0, 1e-5);
}

void test_hnsw_filtered_recall() {
    const std::size_t dim = 24;
    const std::size_t n = 1500;
    const std::size_t nq = 40;
    const std::size_t k = 10;

    tinyann::HnswParams p;
    p.M = 16;
    p.ef_construction = 200;
    p.ef_search = 80;
    p.seed = 42;

    tinyann::Index exact(dim, tinyann::Metric::Cosine);
    tinyann::HnswIndex hnsw(dim, tinyann::Metric::Cosine, p);
    std::mt19937_64 rng(321);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = random_unit_vector(dim, rng);
        exact.add(static_cast<std::int64_t>(i), v);
        hnsw.add(static_cast<std::int64_t>(i), v);
    }
    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }

    // ~50% eligible (even ids)
    auto even = [](std::int64_t id) { return (id % 2) == 0; };
    const double rec_even = hnsw.recall_at_k_vs(exact, queries, k, even, /*ef=*/80);
    std::cout << "recall@10[filtered_even]=" << rec_even << "\n";
    CHECK(rec_even > 0.9);

    // Selective filter (~10% eligible)
    auto mod10 = [](std::int64_t id) { return (id % 10) == 0; };
    const double rec_mod = hnsw.recall_at_k_vs(exact, queries, k, mod10, /*ef=*/120);
    std::cout << "recall@10[filtered_mod10]=" << rec_mod << "\n";
    CHECK(rec_mod > 0.85);

    // Pass-all matches unfiltered recall path
    auto all = [](std::int64_t) { return true; };
    const double rec_all = hnsw.recall_at_k_vs(exact, queries, k, all, /*ef=*/80);
    const double rec_unf = hnsw.recall_at_k_vs(exact, queries, k, /*ef=*/80);
    CHECK_NEAR(rec_all, rec_unf, 1e-12);
}

void test_ivfpq_cosine_scale_invariant() {
    // Exact cosine prefers same direction over longer IP; unnormalized data must not
    // make IVFPQ cosine behave like IP.
    const std::size_t dim = 8;
    tinyann::IvfPqParams p;
    p.nlist = 2;
    p.nprobe = 2;
    p.M = 2;
    p.kmeans_iters = 15;
    p.pq_kmeans_iters = 15;
    p.seed = 1;

    // a: short, aligned with e0; b: long, 45° from e0 — cosine(q,a)=1 > cosine(q,b).
    std::vector<float> a(dim, 0.f);
    a[0] = 0.01f;
    std::vector<float> b(dim, 0.f);
    b[0] = 100.f;
    b[1] = 100.f;
    std::mt19937_64 rng(2);

    std::vector<std::vector<float>> train = {a, b, random_unit_vector(dim, rng),
                                             random_unit_vector(dim, rng),
                                             random_unit_vector(dim, rng),
                                             random_unit_vector(dim, rng)};
    tinyann::IvfPqIndex cos_idx(dim, tinyann::Metric::Cosine, p);
    cos_idx.train(train);
    cos_idx.add(1, a);
    cos_idx.add(2, b);

    tinyann::Index exact(dim, tinyann::Metric::Cosine);
    exact.add(1, a);
    exact.add(2, b);

    std::vector<float> q(dim, 0.f);
    q[0] = 1.f;
    const auto exact_hits = exact.search(q, 2);
    CHECK(exact_hits[0].id == 1);  // cosine prefers short aligned a
    CHECK_NEAR(exact_hits[0].score, 1.0, 1e-5);

    tinyann::Index ip_idx(dim, tinyann::Metric::InnerProduct);
    ip_idx.add(1, a);
    ip_idx.add(2, b);
    const auto ip_hits = ip_idx.search(q, 2);
    CHECK(ip_hits[0].id == 2);  // IP prefers long b

    const auto cos_hits = cos_idx.search(q, 2);
    CHECK(cos_hits.size() == 2);
    CHECK(cos_hits[0].id == 1);  // must match cosine, not IP
    CHECK(cos_hits[0].score > cos_hits[1].score);
}

void test_ivfpq_rejects_bad_m() {
    tinyann::IvfPqParams p;
    p.M = 3;  // 32 % 3 != 0
    bool threw = false;
    try {
        tinyann::IvfPqIndex idx(32, tinyann::Metric::Euclidean, p);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_ivfpq_basic_search_and_filter() {
    const std::size_t dim = 32;
    tinyann::IvfPqParams p;
    p.nlist = 8;
    p.nprobe = 8;
    p.M = 8;
    p.kmeans_iters = 15;
    p.pq_kmeans_iters = 15;
    p.seed = 1;

    tinyann::IvfPqIndex idx(dim, tinyann::Metric::InnerProduct, p);
    std::mt19937_64 rng(9);
    std::vector<std::vector<float>> all;
    for (int i = 0; i < 200; ++i) {
        all.push_back(random_unit_vector(dim, rng));
    }
    idx.train(all);
    CHECK(idx.trained());
    for (int i = 0; i < 200; ++i) {
        idx.add(i, all[static_cast<std::size_t>(i)]);
    }
    CHECK(idx.size() == 200);
    CHECK(idx.code_size() == 8);

    auto r = idx.search(all[0], 5);
    CHECK(r.size() == 5);
    // Top-1 should usually be self for unit vectors + IP with full nprobe.
    CHECK(r[0].id == 0);

    auto none = idx.search(all[0], 5, [](std::int64_t) { return false; });
    CHECK(none.empty());
    auto even = idx.search(all[0], 10, [](std::int64_t id) { return id % 2 == 0; });
    for (const auto& h : even) {
        CHECK((h.id % 2) == 0);
    }
    CHECK(!even.empty());
}

void test_ivfpq_recall() {
    // Moderate compression (M=16 → 16 B/vector on dim=64) + full list probe for quality check.
    const std::size_t dim = 64;
    const std::size_t n = 2500;
    const std::size_t nq = 50;
    const std::size_t k = 10;
    tinyann::IvfPqParams p;
    p.nlist = 40;
    p.nprobe = 40;  // probe all lists → focus on PQ distance quality
    p.M = 16;
    p.kmeans_iters = 25;
    p.pq_kmeans_iters = 25;
    p.seed = 11;

    for (auto metric : {tinyann::Metric::Cosine, tinyann::Metric::Euclidean,
                        tinyann::Metric::InnerProduct}) {
        tinyann::Index exact(dim, metric);
        tinyann::IvfPqIndex ivfpq(dim, metric, p);
        std::mt19937_64 rng(200 + static_cast<unsigned>(metric));
        std::vector<std::vector<float>> all;
        all.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            all.push_back(random_unit_vector(dim, rng));
        }
        ivfpq.train(all);
        for (std::size_t i = 0; i < n; ++i) {
            exact.add(static_cast<std::int64_t>(i), all[i]);
            ivfpq.add(static_cast<std::int64_t>(i), all[i]);
        }
        std::vector<std::vector<float>> queries;
        for (std::size_t i = 0; i < nq; ++i) {
            queries.push_back(random_unit_vector(dim, rng));
        }
        const double rec = ivfpq.recall_at_k_vs(exact, queries, k);
        std::cout << "recall@10[ivfpq_" << tinyann::metric_name(metric) << "]=" << rec << "\n";
        // Residual 8-bit PQ without re-rank is coarser than float IVF; M=16 + full nprobe
        // should still recover a majority of true neighbors on unit-vector synthetic data.
        CHECK(rec > 0.55);
    }
}

void test_ivfpq_save_load() {
    tinyann::IvfPqParams p;
    p.nlist = 10;
    p.nprobe = 5;
    p.M = 4;
    p.seed = 3;
    const std::size_t dim = 16;
    tinyann::IvfPqIndex idx(dim, tinyann::Metric::Cosine, p);
    std::mt19937_64 rng(12);
    std::vector<std::vector<float>> all;
    for (int i = 0; i < 120; ++i) {
        all.push_back(random_unit_vector(dim, rng));
    }
    idx.train(all);
    for (int i = 0; i < 120; ++i) {
        idx.add(i, all[static_cast<std::size_t>(i)]);
    }
    const auto q = random_unit_vector(dim, rng);
    const auto before = idx.search(q, 8);
    const std::string path = temp_path("tinyann_ivfpq.bin");
    idx.save(path);
    auto loaded = tinyann::IvfPqIndex::load(path);
    CHECK(loaded.trained());
    CHECK(loaded.size() == idx.size());
    CHECK(loaded.params().M == idx.params().M);
    CHECK(results_byte_identical(before, loaded.search(q, 8)));
}

void test_ivfpq_refine_improves_recall() {
    const std::size_t dim = 64;
    const std::size_t n = 2000;
    const std::size_t nq = 40;
    const std::size_t k = 10;
    tinyann::IvfPqParams p;
    p.nlist = 40;
    p.nprobe = 12;
    p.M = 8;
    p.kmeans_iters = 20;
    p.pq_kmeans_iters = 20;
    p.seed = 42;
    p.store_raw = true;
    p.nrefine = 0;

    tinyann::Index exact(dim, tinyann::Metric::InnerProduct);
    tinyann::IvfPqIndex ivfpq(dim, tinyann::Metric::InnerProduct, p);
    std::mt19937_64 rng(7);
    std::vector<std::vector<float>> all;
    for (std::size_t i = 0; i < n; ++i) {
        all.push_back(random_unit_vector(dim, rng));
    }
    ivfpq.train(all);
    for (std::size_t i = 0; i < n; ++i) {
        exact.add(static_cast<std::int64_t>(i), all[i]);
        ivfpq.add(static_cast<std::int64_t>(i), all[i]);
    }
    CHECK(ivfpq.stores_raw());

    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < nq; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }

    const double rec_plain = ivfpq.recall_at_k_vs(exact, queries, k, /*nrefine=*/0);
    const double rec_ref = ivfpq.recall_at_k_vs(exact, queries, k, /*nrefine=*/80);
    std::cout << "recall@10[ivfpq_plain]=" << rec_plain << " refine80=" << rec_ref << "\n";
    CHECK(rec_ref + 1e-12 >= rec_plain);
    CHECK(rec_ref > rec_plain + 0.05);
    CHECK(rec_ref > 0.55);

    const auto refined = ivfpq.search(queries[0], k, /*nrefine=*/80);
    CHECK(refined.size() == k);
    for (const auto& hit : refined) {
        std::size_t node = 0;
        while (node < ivfpq.ids().size() && ivfpq.ids()[node] != hit.id) {
            ++node;
        }
        CHECK(node < ivfpq.ids().size());
        const float* raw = ivfpq.raw_data().data() + node * dim;
        CHECK_NEAR(hit.score, tinyann::inner_product(queries[0].data(), raw, dim), 1e-4);
    }

    bool threw = false;
    tinyann::IvfPqParams p2 = p;
    p2.store_raw = false;
    p2.nrefine = 0;
    tinyann::IvfPqIndex no_raw(dim, tinyann::Metric::InnerProduct, p2);
    no_raw.train(all);
    no_raw.add(0, all[0]);
    try {
        (void)no_raw.search(queries[0], 1, /*nrefine=*/10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

/// Build unit vectors with strong axis-aligned correlation (second half = first half).
/// Axis-aligned PQ wastes capacity here; OPQ should help.
std::vector<std::vector<float>> make_correlated_unit_corpus(std::size_t dim, std::size_t n,
                                                            std::mt19937_64& rng) {
    std::vector<std::vector<float>> all;
    all.reserve(n);
    const std::size_t half = dim / 2;
    for (std::size_t i = 0; i < n; ++i) {
        auto v = random_unit_vector(half, rng);
        std::vector<float> full(dim);
        for (std::size_t d = 0; d < half; ++d) {
            full[d] = v[d];
            full[d + half] = v[d];
        }
        float n2 = 0.f;
        for (float x : full) {
            n2 += x * x;
        }
        if (n2 > 0.f) {
            const float inv = 1.f / std::sqrt(n2);
            for (float& x : full) {
                x *= inv;
            }
        }
        all.push_back(std::move(full));
    }
    return all;
}

void test_ivfpq_opq_basic_and_save_load() {
    const std::size_t dim = 32;
    tinyann::IvfPqParams p;
    p.nlist = 8;
    p.nprobe = 8;
    p.M = 8;
    p.kmeans_iters = 12;
    p.pq_kmeans_iters = 12;
    p.use_opq = true;
    p.opq_iters = 4;
    p.seed = 3;

    tinyann::IvfPqIndex idx(dim, tinyann::Metric::InnerProduct, p);
    std::mt19937_64 rng(11);
    auto all = make_correlated_unit_corpus(dim, 300, rng);
    idx.train(all);
    CHECK(idx.uses_opq());
    CHECK(idx.opq_matrix().size() == dim * dim);
    for (int i = 0; i < 300; ++i) {
        idx.add(i, all[static_cast<std::size_t>(i)]);
    }
    auto hits = idx.search(all[0], 5);
    CHECK(hits.size() == 5);
    CHECK(hits[0].id == 0);

    const std::string path = temp_path("tinyann_ivfpq_opq.bin");
    idx.save(path);
    auto loaded = tinyann::IvfPqIndex::load(path);
    CHECK(loaded.uses_opq());
    CHECK(results_byte_identical(hits, loaded.search(all[0], 5)));

    // Without OPQ still trains (identity path, no matrix stored).
    tinyann::IvfPqParams p2 = p;
    p2.use_opq = false;
    tinyann::IvfPqIndex plain(dim, tinyann::Metric::InnerProduct, p2);
    plain.train(all);
    CHECK(!plain.uses_opq());
    plain.add(0, all[0]);
    CHECK(!plain.search(all[0], 1).empty());
}

void test_ivfpq_opq_better_than_pq() {
    // On correlated embeddings, OPQ should not lose to plain PQ and should usually win.
    const std::size_t dim = 64;
    const std::size_t n = 2500;
    const std::size_t nq = 60;
    const std::size_t k = 10;

    tinyann::IvfPqParams base;
    base.nlist = 40;
    base.nprobe = 16;
    base.M = 8;  // coarser codes so rotation can matter
    base.kmeans_iters = 20;
    base.pq_kmeans_iters = 20;
    base.seed = 99;
    base.opq_iters = 8;

    std::mt19937_64 rng(2026);
    auto all = make_correlated_unit_corpus(dim, n, rng);
    std::vector<std::vector<float>> queries;
    queries.reserve(nq);
    for (std::size_t i = 0; i < nq; ++i) {
        // Same correlation structure for queries.
        auto batch = make_correlated_unit_corpus(dim, 1, rng);
        queries.push_back(std::move(batch[0]));
    }

    tinyann::Index exact(dim, tinyann::Metric::InnerProduct);
    for (std::size_t i = 0; i < n; ++i) {
        exact.add(static_cast<std::int64_t>(i), all[i]);
    }

    tinyann::IvfPqParams p_pq = base;
    p_pq.use_opq = false;
    tinyann::IvfPqIndex pq(dim, tinyann::Metric::InnerProduct, p_pq);
    pq.train(all);
    for (std::size_t i = 0; i < n; ++i) {
        pq.add(static_cast<std::int64_t>(i), all[i]);
    }
    CHECK(!pq.uses_opq());

    tinyann::IvfPqParams p_opq = base;
    p_opq.use_opq = true;
    tinyann::IvfPqIndex opq(dim, tinyann::Metric::InnerProduct, p_opq);
    opq.train(all);
    for (std::size_t i = 0; i < n; ++i) {
        opq.add(static_cast<std::int64_t>(i), all[i]);
    }
    CHECK(opq.uses_opq());

    const double rec_pq = pq.recall_at_k_vs(exact, queries, k);
    const double rec_opq = opq.recall_at_k_vs(exact, queries, k);
    std::cout << "recall@10[ivfpq_pq]=" << rec_pq << " opq=" << rec_opq << "\n";

    // OPQ must not be worse than plain PQ on this correlated set (allow tiny float noise).
    CHECK(rec_opq + 1e-9 >= rec_pq);
    // Expect a real gain when axes are highly redundant.
    CHECK(rec_opq > rec_pq + 0.02);
    CHECK(rec_opq > 0.45);
}

void test_ivfpq_remove_update() {
    tinyann::IvfPqParams p;
    p.nlist = 4;
    p.nprobe = 4;
    p.M = 2;
    p.seed = 5;
    tinyann::IvfPqIndex idx(8, tinyann::Metric::InnerProduct, p);
    std::mt19937_64 rng(1);
    std::vector<std::vector<float>> all;
    for (int i = 0; i < 40; ++i) {
        all.push_back(random_unit_vector(8, rng));
    }
    idx.train(all);
    for (int i = 0; i < 40; ++i) {
        idx.add(i, all[static_cast<std::size_t>(i)]);
    }
    CHECK(idx.remove(7));
    CHECK(!idx.contains(7));
    CHECK(idx.size() == 39);
    auto hits = idx.search(all[0], 5);
    for (const auto& h : hits) {
        CHECK(h.id != 7);
    }
    CHECK(idx.update(1, all[2]));
    CHECK(idx.search(all[2], 1).size() == 1);
}

void test_remove_update_persist_roundtrip() {
    tinyann::Index exact(2, tinyann::Metric::Cosine);
    exact.add(1, {1.f, 0.f});
    exact.add(2, {0.f, 1.f});
    exact.add(3, {1.f, 1.f});
    exact.remove(2);
    exact.update(3, {0.f, 1.f});
    const std::string pe = temp_path("tinyann_rm_exact.bin");
    exact.save(pe);
    auto el = tinyann::Index::load(pe);
    CHECK(el.size() == 2);
    CHECK(!el.contains(2));
    CHECK(el.contains(1));
    CHECK(el.contains(3));
    const auto q = std::vector<float>{0.f, 1.f};
    CHECK(results_byte_identical(exact.search(q, 2), el.search(q, 2)));

    tinyann::HnswParams p;
    p.M = 8;
    p.ef_construction = 40;
    p.ef_search = 20;
    p.seed = 9;
    tinyann::HnswIndex h(2, tinyann::Metric::Cosine, p);
    h.add(1, {1.f, 0.f});
    h.add(2, {0.f, 1.f});
    h.add(3, {1.f, 1.f});
    h.add(4, {-1.f, 0.f});
    h.remove(2);
    h.update(1, {0.9f, 0.1f});
    const std::string ph = temp_path("tinyann_rm_hnsw.bin");
    h.save(ph);
    auto hl = tinyann::HnswIndex::load(ph);
    CHECK(hl.size() == 3);
    CHECK(!hl.contains(2));
    CHECK(results_byte_identical(h.search(q, 3), hl.search(q, 3)));
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
    test_simd_matches_scalar();
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
    test_exact_save_load_identical_search();
    test_exact_load_rejects_hnsw_file();
    test_hnsw_save_load_identical_search();
    test_hnsw_load_rejects_exact_file();
    test_hnsw_load_rejects_corrupt_graph();
    test_load_rejects_size_overflow();
    test_parse_vector_text_line_int64_ids();
    test_reject_non_finite();
    test_persistence_all_metrics();
    test_exact_remove_update();
    test_hnsw_remove_update();
    test_hnsw_remove_entry_point();
    test_hnsw_remove_keeps_searchable();
    test_remove_update_persist_roundtrip();
    test_exact_filtered_search();
    test_hnsw_filtered_search_basic();
    test_hnsw_filtered_not_postfilter_topk();
    test_hnsw_filtered_recall();
    test_hnsw_concurrent_search();
    test_ivf_basic_and_filter();
    test_ivf_recall();
    test_ivf_save_load();
    test_ivf_load_rejects_corrupt_lists();
    test_ivf_remove_update();
    test_ivfpq_rejects_bad_m();
    test_ivfpq_basic_search_and_filter();
    test_ivfpq_cosine_scale_invariant();
    test_ivfpq_recall();
    test_ivfpq_save_load();
    test_ivfpq_refine_improves_recall();
    test_ivfpq_opq_basic_and_save_load();
    test_ivfpq_opq_better_than_pq();
    test_ivfpq_remove_update();
    test_sq_quantize_dequantize();
    test_index_sq_search_and_recall();
    test_index_sq_save_load();

    std::cout << "passed=" << g_passed << " failed=" << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
