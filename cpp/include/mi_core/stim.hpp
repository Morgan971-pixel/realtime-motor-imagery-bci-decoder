// Stim validation and marker detection (contract step 4).
#pragma once

#include <cstdint>
#include <optional>

namespace mi {

constexpr double kMaxCode = 2147483647.0;
constexpr std::uint32_t kImageryPhase = 3;
constexpr std::uint32_t kLeft = 1;
constexpr std::uint32_t kRight = 2;

struct Marker {
  std::uint32_t trial;
  std::uint32_t phase;
  std::uint32_t movement;

  bool is_imagery() const { return phase == kImageryPhase && (movement == kLeft || movement == kRight); }
  bool operator==(const Marker&) const = default;
};

/// Check one raw stim value; returns the integer code (0 = no event).
/// Throws std::invalid_argument with the reason, worded as in the Rust core; the decoder
/// rethrows it as StimError with the sample index.
std::uint32_t validate_code(double value);

/// Split a validated nonzero code into trial, phase and movement (integer division).
Marker parse_code(std::uint32_t code);

/// A nonzero code is a new marker unless the previous sample held the same code.
class MarkerTracker {
 public:
  /// Call only with codes that passed validate_code, and only after the whole sample was
  /// validated: it updates the previous code, so a later rejection would leave it changed.
  std::optional<Marker> observe(std::uint32_t code);

 private:
  std::uint32_t prev_ = 0;
};

}  // namespace mi
