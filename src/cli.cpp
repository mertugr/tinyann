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
        << "  " << argv0 << " --dim N --metric METRIC --vectors FILE --query FILE [--k K]\n\n"
        << "Options:\n"
        << "  --dim N           Vector dimension (required)\n"
        << "  --metric METRIC   cosine | euclidean | l2 | inner_product | ip  (default: cosine)\n"
        << "  --vectors FILE    Text file: one vector per line: <id> <f1> <f2> ... <fN>\n"
        << "  --query FILE      Query vector file: either \"<id> <f1> ... <fN>\" or \"<f1> ... <fN>\"\n"
        << "                    If multiple lines, each line is a separate query.\n"
        << "  --k K             Number of nearest neighbors (default: 10)\n"
        << "  -h, --help        Show this help\n\n"
        << "Output (per query): one line per hit: <rank>\\t<id>\\t<score>\n";
}

struct Options {
    std::size_t dim = 0;
    tinyann::Metric metric = tinyann::Metric::Cosine;
    std::string vectors_path;
    std::string query_path;
    std::size_t k = 10;
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
            opt.metric = tinyann::Index::parse_metric(v);
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
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

// Parse a line of floats. If leading_id is true and the line has dim+1 tokens,
// the first token is treated as int64 id.
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

bool load_vectors(const std::string& path, std::size_t dim, tinyann::Index& index) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open vectors file: " << path << "\n";
        return false;
    }
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        // Skip empty / comment lines
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
            // No id in file: use 1-based line number among data lines is awkward;
            // use sequential 0-based insertion index.
            id = static_cast<std::int64_t>(index.size());
        }
        try {
            index.add(id, vec);
        } catch (const std::exception& e) {
            std::cerr << "add failed on line " << line_no << ": " << e.what() << "\n";
            return false;
        }
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
        // Query may optionally start with an ignored id
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

    try {
        tinyann::Index index(opt.dim, opt.metric);
        if (!load_vectors(opt.vectors_path, opt.dim, index)) {
            return 1;
        }

        std::vector<std::vector<float>> queries;
        if (!load_queries(opt.query_path, opt.dim, queries)) {
            return 1;
        }

        std::cerr << "index size=" << index.size()
                  << " dim=" << index.dimension()
                  << " metric=" << tinyann::Index::metric_name(index.metric())
                  << " k=" << opt.k << "\n";

        for (std::size_t qi = 0; qi < queries.size(); ++qi) {
            if (queries.size() > 1) {
                std::cout << "# query " << qi << "\n";
            }
            const auto results = index.search(queries[qi], opt.k);
            for (std::size_t r = 0; r < results.size(); ++r) {
                std::cout << (r + 1) << '\t' << results[r].id << '\t' << results[r].score << '\n';
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
