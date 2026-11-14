// Python bindings: `import mi_core_cpp`, the same API as the Rust core's `mi_core`.
// Every contract rejection (mi::Error) becomes a Python ValueError with the same message.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include "mi_core/csv.hpp"
#include "mi_core/decoder.hpp"
#include "mi_core/error.hpp"
#include "mi_core/model.hpp"

namespace py = pybind11;

namespace {

py::dict decision_dict(const mi::Decision& d) {
  py::dict out;
  out["sample"] = d.sample;
  out["trial"] = d.trial;
  out["true_label"] = d.true_label;
  out["features"] = d.features;
  out["score"] = d.score;
  out["label"] = d.label;
  out["pinned"] = d.pinned;
  return out;
}

/// Any sequence of 8 numbers (list, tuple, NumPy row); a wrong length is a ValueError.
mi::Sample to_sample(const std::vector<double>& values) {
  if (values.size() != mi::kChannels) {
    throw py::value_error("expected a sequence of length 8 (got " + std::to_string(values.size()) + ")");
  }
  mi::Sample s{};
  for (std::size_t c = 0; c < mi::kChannels; ++c) s[c] = values[c];
  return s;
}

}  // namespace

PYBIND11_MODULE(mi_core_cpp, m) {
  m.doc() = "C++ inference core for mi-realtime";
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const mi::Error& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });

  py::class_<mi::Decoder>(m, "Decoder", "Streaming decoder: push one sample at a time.")
      .def(py::init([](const std::string& model_json) { return mi::Decoder(mi::Model::from_json(model_json)); }),
           py::arg("model_json"))
      .def(
          "push",
          [](mi::Decoder& self, const std::vector<double>& sample, double stim) -> std::optional<py::dict> {
            auto d = self.push(to_sample(sample), stim);
            if (!d) return std::nullopt;
            return decision_dict(*d);
          },
          py::arg("sample"), py::arg("stim"),
          "Feed 8 EEG values and the stim code; returns a decision dict when one is due, else None.")
      .def_property_readonly("samples_seen", &mi::Decoder::samples_seen);

  m.def(
      "check_model", [](const std::string& model_json) { mi::Model::from_json(model_json); }, py::arg("model_json"),
      "Validate a model.json text; raises ValueError if it breaks the contract.");

  m.def(
      "decode_file",
      [](const std::string& model_path, const std::string& csv_path) {
        std::vector<mi::Decision> decisions;
        {
          py::gil_scoped_release release;  // other Python threads run while the file decodes
          // like the Rust core: a model error names the model file, everything else the CSV
          const mi::Model model = [&] {
            try {
              return mi::Model::from_path(model_path);
            } catch (const mi::Error& e) {
              throw mi::Error(model_path + ": " + e.what());
            }
          }();
          try {
            decisions = mi::decode(model, mi::read_csv(csv_path));
          } catch (const mi::Error& e) {
            throw mi::Error(csv_path + ": " + e.what());
          }
        }
        py::list out;
        for (const auto& d : decisions) out.append(decision_dict(d));
        return out;
      },
      py::arg("model_path"), py::arg("csv_path"), "Decode a club CSV recording; returns a list of dicts.");
}
