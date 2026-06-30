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
        << "  Build from vectors + query:\n"
        << "    " << argv0 << " --dim N --metric M --vectors FILE --query FILE [options]\n"
        << "  Load saved index + query:\n"
        << "    " << argv0 << " --load PATH --query FILE [--index exact|hnsw] [options]\n\n"
        << "Options:\n"
        << "  --dim N           Vector dimension (required unless --load)\n"
        << "  --metric METRIC   cosine | euclidean | l2 | inner_product | ip  (default: cosine)\n"
        << "  --vectors FILE    Text file: one vector per line: <id> <f1> <f2> ... <fN>\n"
        << "  --query FILE      Query vector file (optional if only --save)\n"
        << "  --k K             Number of nearest neighbors (default: 10)\n"
        << "  --index TYPE      exact | brute | hnsw | approx  (default: exact)\n"
        << "  --ef N            HNSW ef_search candidate list size (default: 64)\n"
        << "  --M N             HNSW M (max links per layer; default: 16)\n"
        << "  --efc N           HNSW ef_construction (default: 200)\n"
        << "  --save PATH       Write built/loaded index to PATH (binary)\n"
        << "  --load PATH       Load index from PATH instead of --vectors\n"
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
    std::string index_type = "exact";
    std::size_t ef = 64;
    std::size_t M = 16;
    std::size_t efc = 200;
    std::string save_path;
    std::string load_path;
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
        } else if (arg == "--save") {
            const char* v = need("--save");
            if (!v) return false;
            opt.save_path = v;
        } else if (arg == "--load") {
            const char* v = need("--load");
            if (!v) return false;
            opt.load_path = v;
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

    if (!use_hnsw(opt.index_type) && !use_exact(opt.index_type)) {
        std::cerr << "unknown --index type: " << opt.index_type
                  << " (use exact|brute|hnsw|approx)\n";
        return 2;
    }

    const bool loading = !opt.load_path.empty();
    const bool from_vectors = !opt.vectors_path.empty();
    if (loading && from_vectors) {
        std::cerr << "use either --load or --vectors, not both\n";
        return 2;
    }
    if (!loading && !from_vectors) {
        std::cerr << "provide --vectors FILE or --load PATH\n";
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
        const bool approx = use_hnsw(opt.index_type);

        if (!approx) {
            tinyann::Index index(1, opt.metric);  // placeholder, replaced below
            if (loading) {
                index = tinyann::Index::load(opt.load_path);
                std::cerr << "loaded exact index from " << opt.load_path << "\n";
            } else {
                index = tinyann::Index(opt.dim, opt.metric);
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
                      << "\n";

            if (!opt.query_path.empty()) {
                std::vector<std::vector<float>> queries;
                if (!load_queries(opt.query_path, index.dimension(), queries)) {
                    return 1;
                }
                for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                    if (queries.size() > 1) {
                        std::cout << "# query " << qi << "\n";
                    }
                    print_hits(index.search(queries[qi], opt.k));
                }
            }
            return 0;
        }

        // HNSW path
        tinyann::HnswIndex hnsw(1, opt.metric);  // placeholder
        tinyann::Index exact_for_recall(1, opt.metric);
        bool have_exact = false;

        if (loading) {
            hnsw = tinyann::HnswIndex::load(opt.load_path);
            std::cerr << "loaded hnsw index from " << opt.load_path << "\n";
            if (opt.ef != 64) {
                hnsw.set_ef_search(opt.ef);
            }
        } else {
            tinyann::HnswParams hp;
            hp.M = opt.M;
            hp.ef_construction = opt.efc;
            hp.ef_search = opt.ef;
            hnsw = tinyann::HnswIndex(opt.dim, opt.metric, hp);
            exact_for_recall = tinyann::Index(opt.dim, opt.metric);
            have_exact = opt.measure_recall;

            LoadedData data;
            if (!load_vectors_file(opt.vectors_path, opt.dim, data)) {
                return 1;
            }
            for (std::size_t i = 0; i < data.ids.size(); ++i) {
                hnsw.add(data.ids[i], data.vectors[i]);
                if (have_exact) {
                    exact_for_recall.add(data.ids[i], data.vectors[i]);
                }
            }
        }

        if (!opt.save_path.empty()) {
            hnsw.save(opt.save_path);
            std::cerr << "saved hnsw index to " << opt.save_path << "\n";
        }

        std::cerr << "index=hnsw size=" << hnsw.size() << " dim=" << hnsw.dimension()
                  << " metric=" << tinyann::metric_name(hnsw.metric()) << " k=" << opt.k
                  << " M=" << hnsw.params().M << " ef=" << hnsw.params().ef_search
                  << " efc=" << hnsw.params().ef_construction << "\n";

        if (!opt.query_path.empty()) {
            std::vector<std::vector<float>> queries;
            if (!load_queries(opt.query_path, hnsw.dimension(), queries)) {
                return 1;
            }
            for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                if (queries.size() > 1) {
                    std::cout << "# query " << qi << "\n";
                }
                print_hits(hnsw.search(queries[qi], opt.k, hnsw.params().ef_search));
            }

            if (opt.measure_recall) {
                if (!have_exact) {
                    std::cerr << "--recall with --load requires rebuilding exact from --vectors; "
                                 "skipping recall\n";
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
