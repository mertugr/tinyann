#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tinyann/tinyann.hpp"

namespace py = pybind11;

namespace {

tinyann::Metric parse_python_metric(const std::string& name) {
    if (name == "cosine") {
        return tinyann::Metric::Cosine;
    }
    throw std::invalid_argument("tinyann.Index: only metric='cosine' is exposed");
}

}  // namespace

PYBIND11_MODULE(tinyann, m) {
    m.doc() = "tinyann exact index (cosine). add / search / remove.";

    py::class_<tinyann::Index>(m, "Index")
        .def(py::init([](std::size_t dim, const std::string& metric) {
                 return tinyann::Index(dim, parse_python_metric(metric));
             }),
             py::arg("dim"), py::arg("metric") = "cosine")
        .def_property_readonly("dim", &tinyann::Index::dimension)
        .def_property_readonly(
            "metric", [](const tinyann::Index& self) { return tinyann::metric_name(self.metric()); })
        .def("__len__", &tinyann::Index::size)
        .def("contains", &tinyann::Index::contains, py::arg("id"))
        .def(
            "add",
            [](tinyann::Index& self, std::int64_t id, const std::vector<float>& vector) {
                if (self.contains(id)) {
                    throw std::invalid_argument("tinyann.Index.add: duplicate id");
                }
                self.add(id, vector);
            },
            py::arg("id"), py::arg("vector"))
        .def("remove", &tinyann::Index::remove, py::arg("id"))
        .def(
            "search",
            [](const tinyann::Index& self, const std::vector<float>& query, std::size_t k,
               const std::optional<std::vector<std::int64_t>>& allow_ids) {
                std::vector<tinyann::SearchResult> hits;
                if (!allow_ids.has_value()) {
                    hits = self.search(query, k);
                } else {
                    const std::unordered_set<std::int64_t> allow(allow_ids->begin(),
                                                                 allow_ids->end());
                    hits = self.search(query, k, [&](std::int64_t id) {
                        return allow.find(id) != allow.end();
                    });
                }
                py::list out;
                for (const auto& hit : hits) {
                    out.append(py::make_tuple(hit.id, hit.score));
                }
                return out;
            },
            py::arg("query"), py::arg("k"), py::arg("allow_ids") = py::none());
}
