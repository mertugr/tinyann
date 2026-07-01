#include "tinyann/tinyann.hpp"
#include "tinyann/text_io.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::cerr
        << "tinyann — small in-memory vector similarity search\n\n"
        << "Usage:\n"
        << "  Build from vectors + query:\n"
        << "    " << argv0 << " --dim N --metric M --vectors FILE --query FILE [options]\n"
        << "  Load saved index + query:\n"
        << "    " << argv0
        << " --load PATH --query FILE [--index exact|hnsw|ivf|ivfpq] [options]\n"
        << "  Benchmark (synthetic data):\n"
        << "    " << argv0 << " --bench --dim N [--n N] [--nq N] [--k K] [--metric M]\n"
        << "                    [--ef N] [--M N] [--efc N] [--nlist N] [--nprobe N]\n"
        << "                    [--seed N] [--warmup N] [--runs N]\n\n"
        << "Options:\n"
        << "  --dim N           Vector dimension (required unless --load)\n"
        << "  --metric METRIC   cosine | euclidean | l2 | inner_product | ip  (default: cosine)\n"
        << "  --vectors FILE    Text file: one vector per line: <id> <f1> <f2> ... <fN>\n"
        << "  --query FILE      Query vector file (optional if only --save)\n"
        << "  --k K             Number of nearest neighbors (default: 10)\n"
        << "  --index TYPE      exact | brute | hnsw | approx | ivf | ivfpq  (default: exact)\n"
        << "  --sq              Store exact index vectors as int8 scalar-quantized codes\n"
        << "                    (only with --index exact; uses IndexSq)\n"
        << "  --ef N            HNSW ef_search candidate list size (default: 64)\n"
        << "  --M N             HNSW M (max links per layer; default: 16)\n"
        << "  --efc N           HNSW ef_construction (default: 200)\n"
        << "  --nlist N         IVF/IVFPQ number of coarse centroids / lists (default: 100)\n"
        << "  --nprobe N        IVF/IVFPQ lists probed at query time (default: 10)\n"
        << "  --pq-m N          IVFPQ number of subquantizers M; dim must be divisible by M\n"
        << "                    (default: 8)\n"
        << "  --pq-iters N      IVFPQ per-subspace k-means iterations (default: 25)\n"
        << "  --opq             IVFPQ: enable Optimized Product Quantization (learned rotation)\n"
        << "  --opq-iters N     IVFPQ: OPQ alternating iterations (default: 10)\n"
        << "  --store-raw       IVFPQ: keep original floats for exact re-rank (more RAM)\n"
        << "  --nrefine N       IVFPQ: re-rank shortlist size (0=off; requires --store-raw)\n"
        << "  --save PATH       Write built/loaded index to PATH (binary)\n"
        << "  --load PATH       Load index from PATH instead of --vectors\n"
        << "  --allow-ids FILE  Filtered search: only ids listed in FILE (one int64 per line)\n"
        << "  --recall          Also run exact search and print mean recall@k vs exact\n"
        << "  --bench           Run benchmark mode (exact vs HNSW vs IVF + recall)\n"
        << "  --n N             Bench: number of base vectors (default: 20000)\n"
        << "  --nq N            Bench: number of queries (default: 200)\n"
        << "  --seed N          Bench: RNG seed (default: 42)\n"
        << "  --warmup N        Bench: warmup query passes (default: 1)\n"
        << "  --runs N          Bench: timed query passes (default: 3)\n"
        << "  -h, --help        Show this help\n\n"
        << "Output (per query): one line per hit: <rank>\\t<id>\\t<score>\n";
}

struct Options {
    std::size_t dim = 0;
    tinyann::Metric metric = tinyann::Metric::Cosine;
    std::string vectors_path;
    std::string query_path;
    std::size_t k = 10;
    std::string index_type = "exact";
    std::size_t ef = 64;
    std::size_t M = 16;
    std::size_t efc = 200;
    std::size_t nlist = 100;
    std::size_t nprobe = 10;
    std::size_t pq_m = 8;
    std::size_t pq_iters = 25;
    bool use_opq = false;
    std::size_t opq_iters = 10;
    bool store_raw = false;
    std::size_t nrefine = 0;
    std::string save_path;
    std::string load_path;
    std::string allow_ids_path;
    bool measure_recall = false;
    bool use_sq = false;
    bool bench = false;
    std::size_t n = 20000;
    std::size_t nq = 200;
    std::uint64_t seed = 42;
    std::size_t warmup = 1;
    std::size_t runs = 3;
    bool help = false;
};

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            opt.help = true;
            return true;
        } else if (arg == "--dim") {
            const char* v = need("--dim");
            if (!v) return false;
            opt.dim = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--metric") {
            const char* v = need("--metric");
            if (!v) return false;
            opt.metric = tinyann::parse_metric(v);
        } else if (arg == "--vectors") {
            const char* v = need("--vectors");
            if (!v) return false;
            opt.vectors_path = v;
        } else if (arg == "--query") {
            const char* v = need("--query");
            if (!v) return false;
            opt.query_path = v;
        } else if (arg == "--k") {
            const char* v = need("--k");
            if (!v) return false;
            opt.k = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--index") {
            const char* v = need("--index");
            if (!v) return false;
            opt.index_type = v;
        } else if (arg == "--ef") {
            const char* v = need("--ef");
            if (!v) return false;
            opt.ef = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--M") {
            const char* v = need("--M");
            if (!v) return false;
            opt.M = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--efc") {
            const char* v = need("--efc");
            if (!v) return false;
            opt.efc = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--nlist") {
            const char* v = need("--nlist");
            if (!v) return false;
            opt.nlist = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--nprobe") {
            const char* v = need("--nprobe");
            if (!v) return false;
            opt.nprobe = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--pq-m") {
            const char* v = need("--pq-m");
            if (!v) return false;
            opt.pq_m = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--pq-iters") {
            const char* v = need("--pq-iters");
            if (!v) return false;
            opt.pq_iters = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--opq") {
            opt.use_opq = true;
        } else if (arg == "--opq-iters") {
            const char* v = need("--opq-iters");
            if (!v) return false;
            opt.opq_iters = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--store-raw") {
            opt.store_raw = true;
        } else if (arg == "--nrefine") {
            const char* v = need("--nrefine");
            if (!v) return false;
            opt.nrefine = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--save") {
            const char* v = need("--save");
            if (!v) return false;
            opt.save_path = v;
        } else if (arg == "--load") {
            const char* v = need("--load");
            if (!v) return false;
            opt.load_path = v;
        } else if (arg == "--allow-ids") {
            const char* v = need("--allow-ids");
            if (!v) return false;
            opt.allow_ids_path = v;
        } else if (arg == "--recall") {
            opt.measure_recall = true;
        } else if (arg == "--sq") {
            opt.use_sq = true;
        } else if (arg == "--bench") {
            opt.bench = true;
        } else if (arg == "--n") {
            const char* v = need("--n");
            if (!v) return false;
            opt.n = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--nq") {
            const char* v = need("--nq");
            if (!v) return false;
            opt.nq = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--seed") {
            const char* v = need("--seed");
            if (!v) return false;
            opt.seed = static_cast<std::uint64_t>(std::stoull(v));
        } else if (arg == "--warmup") {
            const char* v = need("--warmup");
            if (!v) return false;
            opt.warmup = static_cast<std::size_t>(std::stoull(v));
        } else if (arg == "--runs") {
            const char* v = need("--runs");
            if (!v) return false;
            opt.runs = static_cast<std::size_t>(std::stoull(v));
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

struct LoadedData {
    std::vector<std::int64_t> ids;
    std::vector<std::vector<float>> vectors;
};

bool load_vectors_file(const std::string& path, std::size_t dim, LoadedData& out) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open vectors file: " << path << "\n";
        return false;
    }
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        bool has_id = false;
        std::int64_t id = 0;
        std::vector<float> vec;
        if (!tinyann::parse_vector_text_line(line, dim, /*allow_id=*/true, has_id, id, vec)) {
            std::cerr << "bad vector on line " << line_no << " of " << path << "\n";
            return false;
        }
        // Auto-id only when the line has no id column (not when id is negative).
        if (!has_id) {
            id = static_cast<std::int64_t>(out.ids.size());
        }
        out.ids.push_back(id);
        out.vectors.push_back(std::move(vec));
    }
    return true;
}

bool load_queries(const std::string& path, std::size_t dim,
                  std::vector<std::vector<float>>& queries) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open query file: " << path << "\n";
        return false;
    }
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        bool has_id = false;
        std::int64_t id = 0;
        std::vector<float> vec;
        // Optional leading id is accepted and ignored for queries.
        if (!tinyann::parse_vector_text_line(line, dim, /*allow_id=*/true, has_id, id, vec)) {
            std::cerr << "bad query on line " << line_no << " of " << path << "\n";
            return false;
        }
        (void)has_id;
        (void)id;
        queries.push_back(std::move(vec));
    }
    if (queries.empty()) {
        std::cerr << "no queries found in " << path << "\n";
        return false;
    }
    return true;
}

bool use_hnsw(const std::string& type) {
    return type == "hnsw" || type == "approx" || type == "approximate";
}

bool use_ivf(const std::string& type) { return type == "ivf"; }

bool use_ivfpq(const std::string& type) {
    return type == "ivfpq" || type == "ivf_pq" || type == "pq";
}

bool use_exact(const std::string& type) {
    return type == "exact" || type == "brute" || type == "bruteforce" || type == "bf";
}

void print_hits(const std::vector<tinyann::SearchResult>& results) {
    for (std::size_t r = 0; r < results.size(); ++r) {
        std::cout << (r + 1) << '\t' << results[r].id << '\t' << results[r].score << '\n';
    }
}

bool load_id_set(const std::string& path, std::unordered_set<std::int64_t>& out) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open allow-ids file: " << path << "\n";
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::int64_t id = 0;
        if (!(iss >> id)) {
            std::cerr << "bad id line in " << path << ": " << line << "\n";
            return false;
        }
        out.insert(id);
    }
    return true;
}

std::vector<float> random_unit(std::size_t dim, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(dim);
    float n2 = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] = nd(rng);
        n2 += v[i] * v[i];
    }
    if (n2 > 0.f) {
        const float inv = 1.f / std::sqrt(n2);
        for (float& x : v) {
            x *= inv;
        }
    }
    return v;
}

bool same_ids(const std::vector<tinyann::SearchResult>& a,
              const std::vector<tinyann::SearchResult>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id) {
            return false;
        }
    }
    return true;
}

template <typename Fn>
double time_ms(Fn&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int run_bench(const Options& opt) {
    if (opt.dim == 0) {
        std::cerr << "--bench requires --dim\n";
        return 2;
    }
    if (opt.n == 0 || opt.nq == 0 || opt.k == 0 || opt.runs == 0) {
        std::cerr << "--n, --nq, --k, --runs must be > 0\n";
        return 2;
    }

    std::mt19937_64 rng(opt.seed);
    std::vector<std::vector<float>> base;
    base.reserve(opt.n);
    for (std::size_t i = 0; i < opt.n; ++i) {
        base.push_back(random_unit(opt.dim, rng));
    }
    std::vector<std::vector<float>> queries;
    queries.reserve(opt.nq);
    for (std::size_t i = 0; i < opt.nq; ++i) {
        queries.push_back(random_unit(opt.dim, rng));
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== tinyann benchmark ===\n";
    std::cout << "distance_backend=" << tinyann::distance_backend() << "\n";
    std::cout << "metric=" << tinyann::metric_name(opt.metric) << " dim=" << opt.dim
              << " n=" << opt.n << " nq=" << opt.nq << " k=" << opt.k << "\n";
    std::cout << "hnsw: M=" << opt.M << " ef=" << opt.ef << " efc=" << opt.efc << "\n";
    std::cout << "ivf: nlist=" << opt.nlist << " nprobe=" << opt.nprobe << " seed=" << opt.seed
              << "\n";

    // Build exact
    tinyann::Index exact(opt.dim, opt.metric);
    const double exact_build_ms = time_ms([&] {
        for (std::size_t i = 0; i < opt.n; ++i) {
            exact.add(static_cast<std::int64_t>(i), base[i]);
        }
    });

    // Build HNSW
    tinyann::HnswParams hp;
    hp.M = opt.M;
    hp.ef_construction = opt.efc;
    hp.ef_search = opt.ef;
    hp.seed = opt.seed;
    tinyann::HnswIndex hnsw(opt.dim, opt.metric, hp);
    const double hnsw_build_ms = time_ms([&] {
        for (std::size_t i = 0; i < opt.n; ++i) {
            hnsw.add(static_cast<std::int64_t>(i), base[i]);
        }
    });

    // Build IVF (train on base, then add)
    tinyann::IvfParams ip;
    ip.nlist = opt.nlist;
    ip.nprobe = opt.nprobe;
    ip.seed = opt.seed;
    tinyann::IvfIndex ivf(opt.dim, opt.metric, ip);
    const double ivf_build_ms = time_ms([&] {
        ivf.train(base);
        for (std::size_t i = 0; i < opt.n; ++i) {
            ivf.add(static_cast<std::int64_t>(i), base[i]);
        }
    });

    std::cout << "build_exact_ms=" << exact_build_ms << "\n";
    std::cout << "build_hnsw_ms=" << hnsw_build_ms << "\n";
    std::cout << "build_ivf_ms=" << ivf_build_ms << "\n";

    auto run_exact = [&] {
        for (const auto& q : queries) {
            (void)exact.search(q, opt.k);
        }
    };
    auto run_hnsw = [&] {
        for (const auto& q : queries) {
            (void)hnsw.search(q, opt.k, opt.ef);
        }
    };
    auto run_ivf = [&] {
        for (const auto& q : queries) {
            (void)ivf.search(q, opt.k);
        }
    };

    for (std::size_t w = 0; w < opt.warmup; ++w) {
        run_exact();
        run_hnsw();
        run_ivf();
    }

    double exact_ms_total = 0.0;
    double hnsw_ms_total = 0.0;
    double ivf_ms_total = 0.0;
    for (std::size_t r = 0; r < opt.runs; ++r) {
        exact_ms_total += time_ms(run_exact);
        hnsw_ms_total += time_ms(run_hnsw);
        ivf_ms_total += time_ms(run_ivf);
    }
    const double exact_ms = exact_ms_total / static_cast<double>(opt.runs);
    const double hnsw_ms = hnsw_ms_total / static_cast<double>(opt.runs);
    const double ivf_ms = ivf_ms_total / static_cast<double>(opt.runs);
    const double exact_qps = 1000.0 * static_cast<double>(opt.nq) / exact_ms;
    const double hnsw_qps = 1000.0 * static_cast<double>(opt.nq) / hnsw_ms;
    const double ivf_qps = 1000.0 * static_cast<double>(opt.nq) / ivf_ms;

    std::cout << "search_exact_ms=" << exact_ms << " qps=" << exact_qps
              << " latency_us=" << (1000.0 * exact_ms / static_cast<double>(opt.nq)) << "\n";
    std::cout << "search_hnsw_ms=" << hnsw_ms << " qps=" << hnsw_qps
              << " latency_us=" << (1000.0 * hnsw_ms / static_cast<double>(opt.nq))
              << " speedup_vs_exact=" << (exact_ms / hnsw_ms) << "x\n";
    std::cout << "search_ivf_ms=" << ivf_ms << " qps=" << ivf_qps
              << " latency_us=" << (1000.0 * ivf_ms / static_cast<double>(opt.nq))
              << " speedup_vs_exact=" << (exact_ms / ivf_ms) << "x\n";

    const double rec_h = hnsw.recall_at_k_vs(exact, queries, opt.k, opt.ef);
    const double rec_i = ivf.recall_at_k_vs(exact, queries, opt.k);
    std::cout << "recall@" << opt.k << "_hnsw=" << rec_h << "\n";
    std::cout << "recall@" << opt.k << "_ivf=" << rec_i << "\n";

    std::size_t stable_h = 0, stable_i = 0;
    for (const auto& q : queries) {
        if (same_ids(hnsw.search(q, opt.k, opt.ef), hnsw.search(q, opt.k, opt.ef))) {
            ++stable_h;
        }
        if (same_ids(ivf.search(q, opt.k), ivf.search(q, opt.k))) {
            ++stable_i;
        }
    }
    std::cout << "hnsw_id_stable=" << stable_h << "/" << opt.nq
              << (stable_h == opt.nq ? " OK" : " FAIL") << "\n";
    std::cout << "ivf_id_stable=" << stable_i << "/" << opt.nq
              << (stable_i == opt.nq ? " OK" : " FAIL") << "\n";

    if (stable_h != opt.nq || stable_i != opt.nq) {
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        if (!parse_args(argc, argv, opt)) {
            print_usage(argv[0]);
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "argument error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

    if (opt.help || argc == 1) {
        print_usage(argv[0]);
        return opt.help ? 0 : 2;
    }

    std::cerr << "distance_backend=" << tinyann::distance_backend() << "\n";

    if (opt.bench) {
        return run_bench(opt);
    }

    if (!use_hnsw(opt.index_type) && !use_exact(opt.index_type) && !use_ivf(opt.index_type) &&
        !use_ivfpq(opt.index_type)) {
        std::cerr << "unknown --index type: " << opt.index_type
                  << " (use exact|brute|hnsw|approx|ivf|ivfpq)\n";
        return 2;
    }

    const bool loading = !opt.load_path.empty();
    const bool from_vectors = !opt.vectors_path.empty();
    if (loading && from_vectors) {
        std::cerr << "use either --load or --vectors, not both\n";
        return 2;
    }
    if (!loading && !from_vectors) {
        std::cerr << "provide --vectors FILE or --load PATH (or --bench)\n";
        print_usage(argv[0]);
        return 2;
    }
    if (from_vectors && opt.dim == 0) {
        std::cerr << "--dim is required when building from --vectors\n";
        return 2;
    }
    if (opt.query_path.empty() && opt.save_path.empty()) {
        std::cerr << "provide --query FILE and/or --save PATH\n";
        return 2;
    }

    try {
        std::unordered_set<std::int64_t> allow_ids;
        const bool filtered = !opt.allow_ids_path.empty();
        if (filtered) {
            if (!load_id_set(opt.allow_ids_path, allow_ids)) {
                return 1;
            }
            std::cerr << "filter=allow-ids count=" << allow_ids.size() << " from "
                      << opt.allow_ids_path << "\n";
        }
        auto pred = [&](std::int64_t id) { return !filtered || allow_ids.count(id) > 0; };

        if (opt.use_sq && !use_exact(opt.index_type)) {
            std::cerr << "--sq currently applies only to --index exact (flat SQ index)\n";
            return 2;
        }

        if (use_exact(opt.index_type) && opt.use_sq) {
            tinyann::IndexSq index =
                loading ? tinyann::IndexSq::load(opt.load_path)
                        : tinyann::IndexSq(opt.dim, opt.metric);
            if (loading) {
                std::cerr << "loaded sq index from " << opt.load_path << "\n";
            } else {
                LoadedData data;
                if (!load_vectors_file(opt.vectors_path, opt.dim, data)) {
                    return 1;
                }
                for (std::size_t i = 0; i < data.ids.size(); ++i) {
                    index.add(data.ids[i], data.vectors[i]);
                }
            }
            if (!opt.save_path.empty()) {
                index.save(opt.save_path);
                std::cerr << "saved sq index to " << opt.save_path << "\n";
            }
            std::cerr << "index=sq(int8) size=" << index.size() << " dim=" << index.dimension()
                      << " metric=" << tinyann::metric_name(index.metric()) << " k=" << opt.k
                      << (filtered ? " filtered=1" : "") << "\n";
            if (!opt.query_path.empty()) {
                std::vector<std::vector<float>> queries;
                if (!load_queries(opt.query_path, index.dimension(), queries)) {
                    return 1;
                }
                tinyann::Index exact_for_recall(index.dimension(), index.metric());
                if (opt.measure_recall && !loading) {
                    LoadedData data;
                    if (load_vectors_file(opt.vectors_path, opt.dim, data)) {
                        for (std::size_t i = 0; i < data.ids.size(); ++i) {
                            exact_for_recall.add(data.ids[i], data.vectors[i]);
                        }
                    }
                }
                for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                    if (queries.size() > 1) {
                        std::cout << "# query " << qi << "\n";
                    }
                    if (filtered) {
                        print_hits(index.search(queries[qi], opt.k, pred));
                    } else {
                        print_hits(index.search(queries[qi], opt.k));
                    }
                }
                if (opt.measure_recall && exact_for_recall.size() > 0) {
                    const double rec = index.recall_at_k_vs(exact_for_recall, queries, opt.k);
                    std::cerr << "recall@" << opt.k << "=" << rec << " (sq vs float exact)\n";
                }
            }
            return 0;
        }

        if (use_ivfpq(opt.index_type)) {
            LoadedData ivfpq_data;
            if (!loading) {
                if (!load_vectors_file(opt.vectors_path, opt.dim, ivfpq_data)) {
                    return 1;
                }
            }
            tinyann::IvfPqIndex ivfpq = [&]() -> tinyann::IvfPqIndex {
                if (loading) {
                    auto loaded = tinyann::IvfPqIndex::load(opt.load_path);
                    std::cerr << "loaded ivfpq index from " << opt.load_path << "\n";
                    if (opt.nprobe != 10) {
                        loaded.set_nprobe(opt.nprobe);
                    }
                    return loaded;
                }
                tinyann::IvfPqParams ip;
                ip.nlist = opt.nlist;
                ip.nprobe = opt.nprobe;
                ip.M = opt.pq_m;
                ip.pq_kmeans_iters = opt.pq_iters;
                ip.seed = opt.seed;
                ip.use_opq = opt.use_opq;
                ip.opq_iters = opt.opq_iters;
                ip.store_raw = opt.store_raw;
                ip.nrefine = opt.nrefine;
                tinyann::IvfPqIndex built(opt.dim, opt.metric, ip);
                built.train(ivfpq_data.vectors);
                for (std::size_t i = 0; i < ivfpq_data.ids.size(); ++i) {
                    built.add(ivfpq_data.ids[i], ivfpq_data.vectors[i]);
                }
                return built;
            }();
            if (loading && opt.nrefine != 0) {
                ivfpq.set_nrefine(opt.nrefine);
            }
            if (!opt.save_path.empty()) {
                ivfpq.save(opt.save_path);
                std::cerr << "saved ivfpq index to " << opt.save_path << "\n";
            }
            const std::size_t nrefine_use =
                opt.nrefine != 0 ? opt.nrefine : ivfpq.params().nrefine;
            std::cerr << "index=ivfpq size=" << ivfpq.size() << " dim=" << ivfpq.dimension()
                      << " metric=" << tinyann::metric_name(ivfpq.metric()) << " k=" << opt.k
                      << " nlist=" << ivfpq.params().nlist << " nprobe=" << ivfpq.params().nprobe
                      << " M=" << ivfpq.params().M << " code_bytes=" << ivfpq.code_size()
                      << " opq=" << (ivfpq.uses_opq() ? 1 : 0)
                      << " store_raw=" << (ivfpq.stores_raw() ? 1 : 0)
                      << " nrefine=" << nrefine_use << (filtered ? " filtered=1" : "") << "\n";
            if (!opt.query_path.empty()) {
                std::vector<std::vector<float>> queries;
                if (!load_queries(opt.query_path, ivfpq.dimension(), queries)) {
                    return 1;
                }
                tinyann::Index exact_for_recall(ivfpq.dimension(), ivfpq.metric());
                if (opt.measure_recall && !loading) {
                    for (std::size_t i = 0; i < ivfpq_data.ids.size(); ++i) {
                        exact_for_recall.add(ivfpq_data.ids[i], ivfpq_data.vectors[i]);
                    }
                }
                for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                    if (queries.size() > 1) {
                        std::cout << "# query " << qi << "\n";
                    }
                    if (filtered) {
                        print_hits(ivfpq.search(queries[qi], opt.k, nrefine_use, pred));
                    } else {
                        print_hits(ivfpq.search(queries[qi], opt.k, nrefine_use));
                    }
                }
                if (opt.measure_recall && exact_for_recall.size() > 0) {
                    const double rec =
                        filtered
                            ? ivfpq.recall_at_k_vs(exact_for_recall, queries, opt.k, nrefine_use,
                                                   pred)
                            : ivfpq.recall_at_k_vs(exact_for_recall, queries, opt.k, nrefine_use);
                    std::cerr << "recall@" << opt.k << "=" << rec << " (ivfpq vs exact)\n";
                }
            }
            return 0;
        }

        if (use_ivf(opt.index_type)) {
            LoadedData ivf_data;
            if (!loading) {
                if (!load_vectors_file(opt.vectors_path, opt.dim, ivf_data)) {
                    return 1;
                }
            }
            tinyann::IvfIndex ivf = [&]() -> tinyann::IvfIndex {
                if (loading) {
                    auto loaded = tinyann::IvfIndex::load(opt.load_path);
                    std::cerr << "loaded ivf index from " << opt.load_path << "\n";
                    if (opt.nprobe != 10) {
                        loaded.set_nprobe(opt.nprobe);
                    }
                    return loaded;
                }
                tinyann::IvfParams ip;
                ip.nlist = opt.nlist;
                ip.nprobe = opt.nprobe;
                tinyann::IvfIndex built(opt.dim, opt.metric, ip);
                built.train(ivf_data.vectors);
                for (std::size_t i = 0; i < ivf_data.ids.size(); ++i) {
                    built.add(ivf_data.ids[i], ivf_data.vectors[i]);
                }
                return built;
            }();
            if (!opt.save_path.empty()) {
                ivf.save(opt.save_path);
                std::cerr << "saved ivf index to " << opt.save_path << "\n";
            }
            std::cerr << "index=ivf size=" << ivf.size() << " dim=" << ivf.dimension()
                      << " metric=" << tinyann::metric_name(ivf.metric()) << " k=" << opt.k
                      << " nlist=" << ivf.params().nlist << " nprobe=" << ivf.params().nprobe
                      << (filtered ? " filtered=1" : "") << "\n";
            if (!opt.query_path.empty()) {
                std::vector<std::vector<float>> queries;
                if (!load_queries(opt.query_path, ivf.dimension(), queries)) {
                    return 1;
                }
                tinyann::Index exact_for_recall(ivf.dimension(), ivf.metric());
                if (opt.measure_recall && !loading) {
                    LoadedData data;
                    if (load_vectors_file(opt.vectors_path, opt.dim, data)) {
                        for (std::size_t i = 0; i < data.ids.size(); ++i) {
                            exact_for_recall.add(data.ids[i], data.vectors[i]);
                        }
                    }
                }
                for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                    if (queries.size() > 1) {
                        std::cout << "# query " << qi << "\n";
                    }
                    if (filtered) {
                        print_hits(ivf.search(queries[qi], opt.k, pred));
                    } else {
                        print_hits(ivf.search(queries[qi], opt.k));
                    }
                }
                if (opt.measure_recall && exact_for_recall.size() > 0) {
                    const double rec =
                        filtered ? ivf.recall_at_k_vs(exact_for_recall, queries, opt.k, pred)
                                 : ivf.recall_at_k_vs(exact_for_recall, queries, opt.k);
                    std::cerr << "recall@" << opt.k << "=" << rec << " (ivf vs exact)\n";
                }
            }
            return 0;
        }

        const bool approx = use_hnsw(opt.index_type);

        if (!approx) {
            tinyann::Index index =
                loading ? tinyann::Index::load(opt.load_path) : tinyann::Index(opt.dim, opt.metric);
            if (loading) {
                std::cerr << "loaded exact index from " << opt.load_path << "\n";
            } else {
                LoadedData data;
                if (!load_vectors_file(opt.vectors_path, opt.dim, data)) {
                    return 1;
                }
                for (std::size_t i = 0; i < data.ids.size(); ++i) {
                    index.add(data.ids[i], data.vectors[i]);
                }
            }

            if (!opt.save_path.empty()) {
                index.save(opt.save_path);
                std::cerr << "saved exact index to " << opt.save_path << "\n";
            }

            std::cerr << "index=exact size=" << index.size() << " dim=" << index.dimension()
                      << " metric=" << tinyann::metric_name(index.metric()) << " k=" << opt.k
                      << (filtered ? " filtered=1" : "") << "\n";

            if (!opt.query_path.empty()) {
                std::vector<std::vector<float>> queries;
                if (!load_queries(opt.query_path, index.dimension(), queries)) {
                    return 1;
                }
                for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                    if (queries.size() > 1) {
                        std::cout << "# query " << qi << "\n";
                    }
                    if (filtered) {
                        print_hits(index.search(queries[qi], opt.k, pred));
                    } else {
                        print_hits(index.search(queries[qi], opt.k));
                    }
                }
            }
            return 0;
        }

        bool have_exact = false;
        LoadedData hnsw_data;
        if (!loading) {
            if (!load_vectors_file(opt.vectors_path, opt.dim, hnsw_data)) {
                return 1;
            }
        }

        // Construct HNSW at the correct dimension (no dim=1 placeholder).
        // exact_for_recall only when building from vectors and --recall is set.
        tinyann::HnswIndex hnsw = [&]() -> tinyann::HnswIndex {
            if (loading) {
                auto loaded = tinyann::HnswIndex::load(opt.load_path);
                std::cerr << "loaded hnsw index from " << opt.load_path << "\n";
                if (opt.ef != 64) {
                    loaded.set_ef_search(opt.ef);
                }
                return loaded;
            }
            tinyann::HnswParams hp;
            hp.M = opt.M;
            hp.ef_construction = opt.efc;
            hp.ef_search = opt.ef;
            tinyann::HnswIndex built(opt.dim, opt.metric, hp);
            for (std::size_t i = 0; i < hnsw_data.ids.size(); ++i) {
                built.add(hnsw_data.ids[i], hnsw_data.vectors[i]);
            }
            return built;
        }();

        tinyann::Index exact_for_recall(hnsw.dimension(), hnsw.metric());
        if (!loading && opt.measure_recall) {
            have_exact = true;
            for (std::size_t i = 0; i < hnsw_data.ids.size(); ++i) {
                exact_for_recall.add(hnsw_data.ids[i], hnsw_data.vectors[i]);
            }
        }

        if (!opt.save_path.empty()) {
            hnsw.save(opt.save_path);
            std::cerr << "saved hnsw index to " << opt.save_path << "\n";
        }

        std::cerr << "index=hnsw size=" << hnsw.size() << " dim=" << hnsw.dimension()
                  << " metric=" << tinyann::metric_name(hnsw.metric()) << " k=" << opt.k
                  << " M=" << hnsw.params().M << " ef=" << hnsw.params().ef_search
                  << " efc=" << hnsw.params().ef_construction
                  << (filtered ? " filtered=1" : "") << "\n";

        if (!opt.query_path.empty()) {
            std::vector<std::vector<float>> queries;
            if (!load_queries(opt.query_path, hnsw.dimension(), queries)) {
                return 1;
            }
            for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                if (queries.size() > 1) {
                    std::cout << "# query " << qi << "\n";
                }
                if (filtered) {
                    print_hits(hnsw.search(queries[qi], opt.k, hnsw.params().ef_search, pred));
                } else {
                    print_hits(hnsw.search(queries[qi], opt.k, hnsw.params().ef_search));
                }
            }

            if (opt.measure_recall) {
                if (!have_exact) {
                    std::cerr << "--recall with --load requires rebuilding exact from --vectors; "
                                 "skipping recall\n";
                } else if (filtered) {
                    const double rec = hnsw.recall_at_k_vs(exact_for_recall, queries, opt.k, pred,
                                                           hnsw.params().ef_search);
                    std::cerr << "recall@" << opt.k << "=" << rec
                              << " (filtered vs exact, " << queries.size() << " queries)\n";
                } else {
                    const double rec =
                        hnsw.recall_at_k_vs(exact_for_recall, queries, opt.k, hnsw.params().ef_search);
                    std::cerr << "recall@" << opt.k << "=" << rec << " (vs exact, " << queries.size()
                              << " queries)\n";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
