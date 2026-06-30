#include "tinyann/tinyann.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::cerr
        << "tinyann — small in-memory vector similarity search\n\n"
        << "Usage:\n"
        << "  " << argv0 << " --dim N --metric METRIC --vectors FILE --query FILE [options]\n\n"
        << "Options:\n"
        << "  --dim N           Vector dimension (required)\n"
        << "  --metric METRIC   cosine | euclidean | l2 | inner_product | ip  (default: cosine)\n"
        << "  --vectors FILE    Text file: one vector per line: <id> <f1> <f2> ... <fN>\n"
        << "  --query FILE      Query vector file: \"<id> <f1> ... <fN>\" or \"<f1> ... <fN>\"\n"
        << "  --k K             Number of nearest neighbors (default: 10)\n"
        << "  --index TYPE      exact | brute | hnsw | approx  (default: exact)\n"
        << "  --ef N            HNSW ef_search candidate list size (default: 64)\n"
        << "  --M N             HNSW M (max links per layer; default: 16)\n"
        << "  --efc N           HNSW ef_construction (default: 200)\n"
        << "  --recall          Also run exact search and print mean recall@k vs exact\n"
        << "  -h, --help        Show this help\n\n"
        << "Output (per query): one line per hit: <rank>\\t<id>\\t<score>\n";
}

struct Options {
    std::size_t dim = 0;
    tinyann::Metric metric = tinyann::Metric::Cosine;
    std::string vectors_path;
    std::string query_path;
    std::size_t k = 10;
    std::string index_type = "exact";  // exact | hnsw
    std::size_t ef = 64;
    std::size_t M = 16;
    std::size_t efc = 200;
    bool measure_recall = false;
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
        } else if (arg == "--recall") {
            opt.measure_recall = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

bool parse_vector_line(const std::string& line, std::size_t dim, bool allow_id,
                       std::int64_t& id_out, std::vector<float>& vec_out) {
    std::istringstream iss(line);
    std::vector<float> tokens;
    float x;
    while (iss >> x) {
        tokens.push_back(x);
    }
    if (tokens.empty()) {
        return false;
    }
    if (tokens.size() == dim) {
        id_out = -1;
        vec_out = std::move(tokens);
        return true;
    }
    if (allow_id && tokens.size() == dim + 1) {
        id_out = static_cast<std::int64_t>(tokens[0]);
        vec_out.assign(tokens.begin() + 1, tokens.end());
        return true;
    }
    return false;
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
        std::int64_t id = 0;
        std::vector<float> vec;
        if (!parse_vector_line(line, dim, /*allow_id=*/true, id, vec)) {
            std::cerr << "bad vector on line " << line_no << " of " << path << "\n";
            return false;
        }
        if (id < 0) {
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
        std::int64_t id = 0;
        std::vector<float> vec;
        if (!parse_vector_line(line, dim, /*allow_id=*/true, id, vec)) {
            std::cerr << "bad query on line " << line_no << " of " << path << "\n";
            return false;
        }
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

bool use_exact(const std::string& type) {
    return type == "exact" || type == "brute" || type == "bruteforce" || type == "bf";
}

void print_hits(const std::vector<tinyann::SearchResult>& results) {
    for (std::size_t r = 0; r < results.size(); ++r) {
        std::cout << (r + 1) << '\t' << results[r].id << '\t' << results[r].score << '\n';
    }
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

    if (opt.dim == 0 || opt.vectors_path.empty() || opt.query_path.empty()) {
        std::cerr << "--dim, --vectors and --query are required\n";
        print_usage(argv[0]);
        return 2;
    }
    if (!use_hnsw(opt.index_type) && !use_exact(opt.index_type)) {
        std::cerr << "unknown --index type: " << opt.index_type
                  << " (use exact|brute|hnsw|approx)\n";
        return 2;
    }

    try {
        LoadedData data;
        if (!load_vectors_file(opt.vectors_path, opt.dim, data)) {
            return 1;
        }
        std::vector<std::vector<float>> queries;
        if (!load_queries(opt.query_path, opt.dim, queries)) {
            return 1;
        }

        // Always build exact index when measuring recall (or when index is exact).
        tinyann::Index exact(opt.dim, opt.metric);
        for (std::size_t i = 0; i < data.ids.size(); ++i) {
            exact.add(data.ids[i], data.vectors[i]);
        }

        const bool approx = use_hnsw(opt.index_type);

        if (!approx) {
            std::cerr << "index=exact size=" << exact.size() << " dim=" << exact.dimension()
                      << " metric=" << tinyann::metric_name(exact.metric()) << " k=" << opt.k
                      << "\n";
            for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                if (queries.size() > 1) {
                    std::cout << "# query " << qi << "\n";
                }
                print_hits(exact.search(queries[qi], opt.k));
            }
            return 0;
        }

        tinyann::HnswParams hp;
        hp.M = opt.M;
        hp.ef_construction = opt.efc;
        hp.ef_search = opt.ef;
        tinyann::HnswIndex hnsw(opt.dim, opt.metric, hp);
        for (std::size_t i = 0; i < data.ids.size(); ++i) {
            hnsw.add(data.ids[i], data.vectors[i]);
        }

        std::cerr << "index=hnsw size=" << hnsw.size() << " dim=" << hnsw.dimension()
                  << " metric=" << tinyann::metric_name(hnsw.metric()) << " k=" << opt.k
                  << " M=" << opt.M << " ef=" << opt.ef << " efc=" << opt.efc << "\n";

        for (std::size_t qi = 0; qi < queries.size(); ++qi) {
            if (queries.size() > 1) {
                std::cout << "# query " << qi << "\n";
            }
            print_hits(hnsw.search(queries[qi], opt.k, opt.ef));
        }

        if (opt.measure_recall) {
            const double rec = hnsw.recall_at_k_vs(exact, queries, opt.k, opt.ef);
            std::cerr << "recall@" << opt.k << "=" << rec << " (vs exact, " << queries.size()
                      << " queries)\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
