#pragma once

#include <cstdint>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace tinyann {

/// Parse one text vector line used by the CLI / sample data files.
///
/// Formats (whitespace-separated):
///   - `f1 … fN`           — no id (`has_id_out == false`); N must equal `dim`
///   - `id f1 … fN`        — leading **integer** id (`has_id_out == true`); N must equal `dim`
///
/// The id is parsed as a full-token `int64` (not float). Vector components are `float`.
/// Returns false on empty input, wrong arity, non-integer id, or non-float components.
inline bool parse_vector_text_line(const std::string& line, std::size_t dim, bool allow_id,
                                   bool& has_id_out, std::int64_t& id_out,
                                   std::vector<float>& vec_out) {
    has_id_out = false;
    id_out = 0;
    vec_out.clear();

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(std::move(tok));
    }
    if (tokens.empty() || dim == 0) {
        return false;
    }

    auto parse_float_token = [](const std::string& s, float& out) -> bool {
        if (s.empty()) {
            return false;
        }
        std::size_t idx = 0;
        try {
            out = std::stof(s, &idx);
        } catch (...) {
            return false;
        }
        return idx == s.size();
    };

    auto parse_int64_token = [](const std::string& s, std::int64_t& out) -> bool {
        if (s.empty()) {
            return false;
        }
        std::size_t idx = 0;
        try {
            // Reject floats like "1.0" / "1e3": entire token must be a base-10 integer.
            out = static_cast<std::int64_t>(std::stoll(s, &idx, 10));
        } catch (...) {
            return false;
        }
        return idx == s.size();
    };

    std::size_t float_begin = 0;
    if (tokens.size() == dim) {
        has_id_out = false;
        float_begin = 0;
    } else if (allow_id && tokens.size() == dim + 1) {
        if (!parse_int64_token(tokens[0], id_out)) {
            return false;
        }
        has_id_out = true;
        float_begin = 1;
    } else {
        return false;
    }

    vec_out.resize(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        if (!parse_float_token(tokens[float_begin + i], vec_out[i])) {
            vec_out.clear();
            return false;
        }
    }
    return true;
}

}  // namespace tinyann
