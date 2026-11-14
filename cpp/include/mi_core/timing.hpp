// Seconds to samples, exactly as the contract writes it: floor(x + 0.5).
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mi {

/// `seconds` and `fs` must already be validated: finite, seconds >= 0, fs > 0. A result too large
/// for size_t saturates, like Rust's `as usize` (the model loader caps windows at 2^20 samples first).
inline std::size_t to_samples(double seconds, double fs) {
  assert(std::isfinite(seconds) && seconds >= 0.0 && std::isfinite(fs) && fs > 0.0);
  const double n = std::floor(seconds * fs + 0.5);
  if (!(n >= 0.0)) return 0;
  if (n >= 18446744073709551616.0) return std::numeric_limits<std::size_t>::max();  // 2^64
  return static_cast<std::size_t>(n);
}

}  // namespace mi
