// mi_replay MODEL CSV [--chunk N] [--repeat N] [--paced]
//
// Replays a recording through the C++ core and prints one JSON summary line
// (the format and argument grammar shared with `python -m mi_ref.replay` and the
// Rust `mi-replay`; see docs/replay.md).
#include <algorithm>
#include <charconv>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "mi_core/csv.hpp"
#include "mi_core/model.hpp"
#include "mi_core/replay.hpp"

namespace {

constexpr const char* kUsage = "usage: mi_replay MODEL CSV [--chunk N] [--repeat N] [--paced]";

struct Run {
  std::string model;
  std::string csv;
  mi::ReplayOptions opts;
};
struct Help {};

/// A count: ASCII decimal digits, from 1 to 2^64 - 1 (and within size_t).
std::optional<std::size_t> parse_count(std::string_view s) {
  if (s.empty() || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) return std::nullopt;
  std::size_t v = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc() || end != s.data() + s.size() || v < 1) return std::nullopt;
  return v;
}

std::optional<std::variant<Run, Help>> parse_args(int argc, char** argv) {
  Run a;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    // `--chunk=5` is the same as `--chunk 5`
    std::string_view flag = arg;
    std::optional<std::string_view> inline_value;
    if (const auto eq = arg.find('='); arg.starts_with("--") && eq != std::string_view::npos) {
      flag = arg.substr(0, eq);
      inline_value = arg.substr(eq + 1);
    }
    if (flag == "--chunk" || flag == "--repeat") {
      std::string_view text;
      if (inline_value) {
        text = *inline_value;
      } else if (i + 1 < argc) {
        text = argv[++i];
      } else {
        return std::nullopt;
      }
      const auto v = parse_count(text);
      if (!v) return std::nullopt;
      (flag == "--chunk" ? a.opts.chunk : a.opts.repeat) = *v;
    } else if (flag == "--paced" && !inline_value) {
      a.opts.paced = true;
    } else if ((flag == "-h" || flag == "--help") && !inline_value) {
      return Help{};
    } else if (flag.starts_with('-') && flag.size() > 1) {
      return std::nullopt;  // unknown option
    } else {
      pos.emplace_back(arg);
    }
  }
  if (pos.size() != 2) return std::nullopt;
  a.model = pos[0];
  a.csv = pos[1];
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const auto parsed = parse_args(argc, argv);
  if (!parsed) {
    std::cerr << kUsage << "\n";
    return 2;
  }
  if (std::holds_alternative<Help>(*parsed)) {
    std::cout << kUsage << "\n";
    return 0;
  }
  const Run& args = std::get<Run>(*parsed);
  std::optional<mi::Model> model;
  try {
    model = mi::Model::from_path(args.model);
  } catch (const std::exception& e) {
    std::cerr << args.model << ": " << e.what() << "\n";
    return 1;
  }
  std::vector<mi::Row> rows;
  try {
    rows = mi::read_csv(args.csv);
  } catch (const std::exception& e) {
    std::cerr << args.csv << ": " << e.what() << "\n";
    return 1;
  }
  if (!rows.empty() && args.opts.repeat > std::numeric_limits<std::size_t>::max() / rows.size()) {
    std::cerr << kUsage << "\n--repeat " << args.opts.repeat << " is too large for " << rows.size() << " samples\n";
    return 2;
  }
  mi::ReplayResult r;
  try {
    mi::SteadyClock clock;
    r = mi::replay(*model, rows, args.opts, clock);
  } catch (const std::exception& e) {
    std::cerr << args.csv << ": " << e.what() << "\n";
    return 1;
  }
  nlohmann::json decisions = nlohmann::json::array();
  for (const auto& d : r.decisions) decisions.push_back({d.sample, d.label});
  const nlohmann::json summary = {
      {"impl", "cpp"},
      {"chunk", args.opts.chunk},
      {"paced", args.opts.paced},
      {"repeat", args.opts.repeat},
      {"samples", r.samples},
      {"elapsed_ns", r.elapsed_ns},
      {"decisions", decisions},
      {"latencies_ns", r.latencies_ns},
  };
  std::cout << summary.dump() << "\n";
  return std::cout.flush() ? 0 : 1;  // a closed pipe is an error, not success
}
