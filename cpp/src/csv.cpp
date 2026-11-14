#include "mi_core/csv.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>

#include "mi_core/error.hpp"

namespace mi {
namespace {

constexpr std::array<std::string_view, 10> kHeader{"timestamps", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "stim"};

std::string_view trim(std::string_view s) {
  const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
  while (!s.empty() && space(s.front())) s.remove_prefix(1);
  while (!s.empty() && space(s.back())) s.remove_suffix(1);
  return s;
}

std::vector<std::string_view> split(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      fields.push_back(trim(line.substr(start, i - start)));
      start = i + 1;
    }
  }
  return fields;
}

/// The value of a decimal literal that from_chars matched but reported out of range. The standard
/// leaves from_chars' output unmodified then (libc++ fills it in anyway), so decide from the text:
/// a positive decimal exponent overflowed to infinity, anything else underflowed to zero.
double out_of_range(std::string_view field) {
  const bool negative = field.front() == '-';
  if (negative) field.remove_prefix(1);
  const std::size_t e = field.find_first_of("eE");
  const std::string_view mantissa = field.substr(0, e);
  const std::size_t point = std::min(mantissa.find('.'), mantissa.size());
  const std::size_t first = mantissa.find_first_not_of("0.");  // exists: zero is never out of range
  long long exp10 = first < point ? static_cast<long long>(point - first) - 1
                                  : -static_cast<long long>(first - point);
  if (e != std::string_view::npos) {
    std::string_view x = field.substr(e + 1);
    const bool down = x.front() == '-';
    if (x.front() == '+' || x.front() == '-') x.remove_prefix(1);
    long long n = 0;
    for (const char c : x) n = std::min(n * 10 + (c - '0'), 1000000LL);  // saturate huge exponents
    exp10 += down ? -n : n;
  }
  const double magnitude = exp10 > 0 ? std::numeric_limits<double>::infinity() : 0.0;
  return negative ? -magnitude : magnitude;
}

/// Decimal or exponent notation, optional leading '+', inf/infinity/nan in any case.
/// Out-of-range values become +-inf or +-0, as in Python and Rust; the decoder rejects non-finite ones.
bool to_double(std::string_view field, double& out) {
  if (!field.empty() && field.front() == '+') {
    field.remove_prefix(1);
    if (!field.empty() && (field.front() == '+' || field.front() == '-')) return false;
  }
  if (field.empty() || field.find('(') != std::string_view::npos) return false;  // no nan(...) payloads
  const char* end = field.data() + field.size();
  const auto [ptr, ec] = std::from_chars(field.data(), end, out, std::chars_format::general);
  if (ptr != end) return false;
  if (ec == std::errc::result_out_of_range) out = out_of_range(field);
  return ec == std::errc() || ec == std::errc::result_out_of_range;
}

}  // namespace

std::vector<Row> parse_csv(std::string_view text) {
  if (text.substr(0, 3) == "\xEF\xBB\xBF") text.remove_prefix(3);  // UTF-8 byte-order mark
  if (text.empty()) throw CsvError("empty file");
  std::vector<Row> rows;
  std::size_t line_no = 0;
  bool header_seen = false;
  while (!text.empty() || !header_seen) {
    const std::size_t nl = text.find('\n');
    const std::string_view line = text.substr(0, nl);
    text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
    ++line_no;
    if (!header_seen) {
      const auto names = split(line);
      if (names.size() != kHeader.size() || !std::equal(names.begin(), names.end(), kHeader.begin())) {
        throw CsvError("unexpected columns on line 1: \"" + std::string(line.substr(0, 120)) + "\"");
      }
      header_seen = true;
      continue;
    }
    if (trim(line).empty()) continue;
    const auto fields = split(line);
    if (fields.size() != kHeader.size()) {
      throw CsvError("line " + std::to_string(line_no) + ": expected 10 columns, got " + std::to_string(fields.size()));
    }
    std::array<double, 10> v{};
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (!to_double(fields[i], v[i])) {
        throw CsvError("line " + std::to_string(line_no) + ", column " + std::string(kHeader[i]) +
                       ": not a number: \"" + std::string(fields[i]) + "\"");
      }
    }
    Row row{};
    std::copy(v.begin() + 1, v.begin() + 9, row.sample.begin());
    row.stim = v[9];
    rows.push_back(row);
  }
  if (rows.empty()) throw CsvError("no samples");
  return rows;
}

std::vector<Row> read_csv(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) throw CsvError("cannot read " + path + ": is a directory");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw CsvError("cannot read " + path);
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (in.bad()) throw CsvError("cannot read " + path);
  return parse_csv(text);
}

}  // namespace mi
