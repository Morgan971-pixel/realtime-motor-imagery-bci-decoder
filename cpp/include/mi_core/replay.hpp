// Replay a recording as if a headset were streaming it, timing every decision.
//
// The same loop as mi_ref.replay and the Rust core (docs/replay.md): fast mode hands each
// chunk over immediately; paced mode first sleeps until the chunk's last sample is due at
// the model's sampling rate. A decision's latency runs from its chunk's handover (when the
// loop is awake and starts pushing) to the moment push returns it.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mi_core/csv.hpp"
#include "mi_core/decoder.hpp"
#include "mi_core/model.hpp"

namespace mi {

/// Monotonic time in nanoseconds, injectable so tests can replay with a fake clock.
class Clock {
 public:
  virtual ~Clock() = default;
  virtual std::uint64_t now_ns() = 0;
  virtual void sleep_until_ns(std::uint64_t target) = 0;
};

/// Monotonic nanoseconds since the clock was created.
class SteadyClock final : public Clock {
 public:
  std::uint64_t now_ns() override;
  void sleep_until_ns(std::uint64_t target) override;

 private:
  std::chrono::steady_clock::time_point origin_ = std::chrono::steady_clock::now();
};

struct ReplayOptions {
  std::size_t chunk = 25;   ///< samples handed over together
  bool paced = false;       ///< deliver chunks at the model's sampling rate
  std::size_t repeat = 1;   ///< play the recording this many times back to back
};

struct ReplayResult {
  std::vector<Decision> decisions;
  std::vector<std::uint64_t> latencies_ns;  ///< one per decision
  std::size_t samples = 0;
  std::uint64_t elapsed_ns = 0;
};

/// Throws std::invalid_argument if chunk or repeat is 0; invalid rows throw the decoder's error.
ReplayResult replay(const Model& model, const std::vector<Row>& rows, const ReplayOptions& opts, Clock& clock);

}  // namespace mi
