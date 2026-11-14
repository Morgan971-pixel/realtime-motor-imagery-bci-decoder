// Shared test helpers: golden paths, contract tolerances, manifest values.
#pragma once

#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mi_core/csv.hpp"
#include "mi_core/model.hpp"

namespace test {

constexpr double kFilterAbs = 1e-9;
constexpr double kFeatureRel = 1e-9;
constexpr double kScoreAbs = 1e-7;

inline std::string golden(const std::string& name) { return std::string(MI_GOLDEN_DIR) + "/" + name; }

inline std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);  // a missing fixture must not pass as a json error
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

inline nlohmann::json read_json(const std::string& name) { return nlohmann::json::parse(read_text(golden(name))); }

inline const mi::Model& model() {
  static const mi::Model m = mi::Model::from_path(golden("model.json"));
  return m;
}

/// Numbers stay numbers; the strings "nan", "inf", "-inf" stand for non-finite values.
inline double to_double(const nlohmann::json& v) {
  if (v.is_number()) return v.get<double>();
  const std::string s = v.get<std::string>();
  if (s == "nan") return std::numeric_limits<double>::quiet_NaN();
  return s == "-inf" ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
}

inline std::vector<mi::Row> manifest_rows(const nlohmann::json& rows) {
  std::vector<mi::Row> out;
  for (const auto& r : rows) {
    mi::Row row{};
    for (std::size_t c = 0; c < mi::kChannels; ++c) row.sample[c] = to_double(r[c]);
    row.stim = to_double(r[8]);
    out.push_back(row);
  }
  return out;
}

/// Relative check for values well away from 0 (features). Not for scores, which can be ~0.
inline bool close_rel(double got, double want, double rel) {
  return std::abs(got - want) <= rel * std::max(std::abs(want), std::numeric_limits<double>::min());  // as Rust's assert_rel
}

}  // namespace test
