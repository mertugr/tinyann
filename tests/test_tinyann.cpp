#include "tinyann/tinyann.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
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
    // query · v = score
    idx.add(1, {1.f, 0.f});   // score 1
    idx.add(2, {0.5f, 0.f});  // score 0.5
    idx.add(3, {0.f, 1.f});   // score 0
    idx.add(4, {-1.f, 0.f});  // score -1
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
    // query = (1, 0)
    idx.add(1, {1.f, 0.f});   // dist 0
    idx.add(2, {1.f, 1.f});   // dist 1
    idx.add(3, {0.f, 0.f});   // dist 1
    idx.add(4, {4.f, 0.f});   // dist 3
    auto r = idx.search({1.f, 0.f}, 3);
    CHECK(r.size() == 3);
    CHECK(r[0].id == 1);
    CHECK_NEAR(r[0].score, 0.0, 1e-5);
    // ids 2 and 3 both distance 1; tie-break by smaller id
    CHECK(r[1].id == 2);
    CHECK_NEAR(r[1].score, 1.0, 1e-5);
    CHECK(r[2].id == 3);
    CHECK_NEAR(r[2].score, 1.0, 1e-5);
}

void test_cosine_ranking() {
    tinyann::Index idx(2, tinyann::Metric::Cosine);
    // query = (1, 0)
    idx.add(1, {2.f, 0.f});   // cosine 1
    idx.add(2, {1.f, 1.f});   // cosine 1/sqrt(2) ≈ 0.707
    idx.add(3, {0.f, 1.f});   // cosine 0
    idx.add(4, {-1.f, 0.f});  // cosine -1
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
    idx.add(1, {0.f, 0.f, 0.f});  // zero stored vector
    idx.add(2, {1.f, 0.f, 0.f});
    // Query zero vector: cosine with anything is 0 by our convention
    auto r0 = idx.search({0.f, 0.f, 0.f}, 2);
    CHECK(r0.size() == 2);
    CHECK_NEAR(r0[0].score, 0.0, 1e-6);
    CHECK_NEAR(r0[1].score, 0.0, 1e-6);

    // Non-zero query vs zero vector -> score 0, non-zero vs itself-direction -> 1
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
    // Both scores 0; tie-break by id
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
    // Both have id 7; higher score first
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

    std::cout << "passed=" << g_passed << " failed=" << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
