#include "mi_core/model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>

#include "mi_core/error.hpp"
#include "mi_core/json_strict.hpp"
#include "mi_core/timing.hpp"

namespace mi {
namespace {

using json = nlohmann::json;
constexpr std::array<const char*, kChannels> kChannelNames{"Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"};

[[noreturn]] void fail(const char* rule, const std::string& message) { throw ModelError(rule, message); }

const json& object(const json& v, const std::string& where) {
  if (!v.is_object()) fail("type", where + " must be a JSON object");
  return v;
}

const json& array(const json& v, const std::string& where) {
  if (!v.is_array()) fail("type", where + " must be a JSON array");
  return v;
}

/// Exactly these keys, plus any of `optional`: an unknown or a missing one is a "keys" error.
void exact_keys(const json& obj, std::initializer_list<const char*> keys, const std::string& where,
                std::initializer_list<const char*> optional = {}) {
  const auto known = [&](const std::string& key) {
    const auto is = [&](const char* k) { return key == k; };
    return std::any_of(keys.begin(), keys.end(), is) || std::any_of(optional.begin(), optional.end(), is);
  };
  for (const auto& item : obj.items()) {
    if (!known(item.key())) {
      fail("keys", where + ": unknown key \"" + item.key() + "\"");
    }
  }
  for (const char* k : keys) {
    if (!obj.contains(k)) fail("keys", where + ": missing key \"" + k + "\"");
  }
}

double number(const json& v, const std::string& where) {
  if (!v.is_number()) fail("type", where + " must be a number");  // booleans are not numbers here
  return v.get<double>();
}

long long integer(const json& v, const std::string& where) {
  if (!v.is_number_integer()) fail("type", where + " must be a JSON integer");
  return v.get<long long>();
}

template <std::size_t N>
std::vector<std::array<double, N>> rows(const json& v, const std::string& where) {
  std::vector<std::array<double, N>> out;
  for (const json& row : array(v, where)) {
    array(row, where);
    if (row.size() != N) fail("shape", where + " rows need " + std::to_string(N) + " numbers");
    std::array<double, N> r{};
    for (std::size_t i = 0; i < N; ++i) r[i] = number(row[i], where);
    out.push_back(r);
  }
  return out;
}

bool all_finite(const std::vector<double>& v) {
  return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

}  // namespace

Model Model::from_json(std::string_view text) {
  const json doc = parse_strict(text);
  object(doc, "model");
  exact_keys(doc, {"schema_version", "fs", "channels", "offset", "window", "bands", "lda", "labels"}, "model",
             {"rail_abs"});
  const json& window = object(doc["window"], "window");
  exact_keys(window, {"start_s", "len_s"}, "window");
  const json& lda = object(doc["lda"], "lda");
  exact_keys(lda, {"w", "b"}, "lda");
  const json& labels = object(doc["labels"], "labels");
  exact_keys(labels, {"score_le_0", "score_gt_0"}, "labels");

  if (integer(doc["schema_version"], "schema_version") != 1) fail("value", "unsupported schema_version");
  if (!doc["offset"].is_string()) fail("type", "offset must be a string");
  if (doc["offset"].get<std::string>() != "first_sample") fail("value", "unsupported offset rule");
  const long long le = integer(labels["score_le_0"], "labels.score_le_0");
  const long long gt = integer(labels["score_gt_0"], "labels.score_gt_0");
  if (le != 1 || gt != 2) fail("value", "labels must be {\"score_le_0\": 1, \"score_gt_0\": 2}");

  std::vector<std::string> channels;
  for (const json& c : array(doc["channels"], "channels")) {
    if (!c.is_string()) fail("type", "channel names must be strings");
    channels.push_back(c.get<std::string>());
  }
  if (!std::equal(channels.begin(), channels.end(), kChannelNames.begin(), kChannelNames.end())) {
    fail("value", "channels must be Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 in that order");
  }

  Model m;
  m.fs_ = number(doc["fs"], "fs");
  if (!(std::isfinite(m.fs_) && m.fs_ > 0.0)) fail("value", "fs must be finite and > 0");
  m.window_start_s_ = number(window["start_s"], "window.start_s");
  m.window_len_s_ = number(window["len_s"], "window.len_s");
  const double start = m.window_start_s_, len = m.window_len_s_;
  if (!(std::isfinite(start) && std::isfinite(len) && start >= 0.0 && len > 0.0)) {
    fail("value", "window needs start_s >= 0 and len_s > 0, both finite");
  }
  for (const auto& [name, seconds] : {std::pair{"start_s", start}, std::pair{"len_s", len}}) {
    const double span = seconds * m.fs_;  // checked before any conversion to an integer
    if (!std::isfinite(span) || span > kMaxWindowSamples) {
      fail("value", std::string("window ") + name + " spans more than 1048576 samples");
    }
  }
  if (to_samples(len, m.fs_) < 2) fail("value", "window must span at least 2 samples");

  const json& bands = array(doc["bands"], "bands");
  if (bands.empty()) fail("shape", "need at least one band");
  std::size_t csp_rows = 0;
  for (std::size_t i = 0; i < bands.size(); ++i) {
    const std::string where = "bands[" + std::to_string(i) + "]";
    const json& b = object(bands[i], where);
    exact_keys(b, {"low_hz", "high_hz", "sos", "csp"}, where);
    const double lo = number(b["low_hz"], where + ".low_hz");
    const double hi = number(b["high_hz"], where + ".high_hz");
    Band band{rows<6>(b["sos"], where + ".sos"), rows<kChannels>(b["csp"], where + ".csp")};
    if (band.sos.empty()) fail("shape", where + ": needs at least one filter section");
    if (i == 0) csp_rows = band.csp.size();
    if (band.csp.size() != csp_rows || csp_rows == 0 || csp_rows % 2 != 0) {
      fail("shape", where + ": every band needs the same even, nonzero number of CSP rows");
    }
    if (!(std::isfinite(lo) && std::isfinite(hi) && 0.0 < lo && lo < hi && hi < m.fs_ / 2.0)) {
      fail("value", where + " must satisfy 0 < low < high < fs/2");
    }
    for (std::size_t k = 0; k < band.sos.size(); ++k) {
      for (double v : band.sos[k]) {
        if (!std::isfinite(v)) fail("value", where + ": sos must be finite");
      }
      if (band.sos[k][3] != 1.0) fail("value", where + ", section " + std::to_string(k) + ": a0 must equal 1");
    }
    for (const Sample& row : band.csp) {
      for (double v : row) {
        if (!std::isfinite(v)) fail("value", where + ": csp must be finite");
      }
    }
    m.bands_.push_back(std::move(band));
  }

  for (const json& w : array(lda["w"], "lda.w")) m.lda_w_.push_back(number(w, "lda.w"));
  m.lda_b_ = number(lda["b"], "lda.b");
  if (m.lda_w_.size() != m.bands_.size() * csp_rows) {
    fail("shape", "lda.w must have " + std::to_string(m.bands_.size() * csp_rows) + " numbers");
  }
  if (!all_finite(m.lda_w_) || !std::isfinite(m.lda_b_)) fail("value", "lda values must be finite");

  if (doc.contains("rail_abs")) {  // optional; null and booleans are not numbers
    const double rail = number(doc["rail_abs"], "rail_abs");
    if (!(std::isfinite(rail) && rail > 0.0)) fail("value", "rail_abs must be finite and > 0");
    m.rail_abs_ = rail;
  }
  for (std::size_t c = 0; c < kChannels; ++c) {
    const bool used = std::any_of(m.bands_.begin(), m.bands_.end(), [&](const Band& b) {
      return std::any_of(b.csp.begin(), b.csp.end(), [&](const Sample& row) { return row[c] != 0.0; });
    });
    if (used) m.used_channels_.push_back(c);
  }
  return m;
}

Model Model::from_path(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ModelError("json", "cannot read " + path);
  std::ostringstream text;
  text << in.rdbuf();
  return from_json(text.str());
}

std::size_t Model::window_len() const { return to_samples(window_len_s_, fs_); }
std::size_t Model::window_start() const { return to_samples(window_start_s_, fs_); }

}  // namespace mi
