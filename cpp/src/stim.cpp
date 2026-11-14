#include "mi_core/stim.hpp"

#include <cassert>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mi {
namespace {

/// Shortest round-trip digits in plain notation, as Rust prints an f64 (131.5, -1, 2147483648).
std::string plain(double value) {
  char buf[400];  // the largest finite double has 309 integer digits
  const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, value, std::chars_format::fixed);
  assert(ec == std::errc());
  return std::string(buf, end);
}

}  // namespace

std::uint32_t validate_code(double value) {
  if (!std::isfinite(value)) throw std::invalid_argument("NaN or infinite value");
  if (value < 0.0) throw std::invalid_argument("negative value " + plain(value));
  if (value > kMaxCode) throw std::invalid_argument("value " + plain(value) + " above 2147483647");
  if (value != std::floor(value)) throw std::invalid_argument("non-integer value " + plain(value));
  const auto code = static_cast<std::uint32_t>(value);
  if (code != 0 && code < 100) {
    throw std::invalid_argument("code " + std::to_string(code) + " below 100 (trial >= 1 required)");
  }
  return code;
}

Marker parse_code(std::uint32_t code) {
  assert(code >= 100 && "parse_code needs a validated nonzero code");
  return Marker{code / 100, (code / 10) % 10, code % 10};
}

std::optional<Marker> MarkerTracker::observe(std::uint32_t code) {
  const bool is_new = code != 0 && code != prev_;
  prev_ = code;
  if (!is_new) return std::nullopt;
  return parse_code(code);
}

}  // namespace mi
