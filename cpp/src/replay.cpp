#include "mi_core/replay.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace mi {

namespace {
/// Paced sleep targets stop growing here (2^63 - 1 ns, about 292 years), as in the Rust core.
constexpr std::uint64_t kMaxDueNs = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
}  // namespace

std::uint64_t SteadyClock::now_ns() {
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - origin_);
  return static_cast<std::uint64_t>(ns.count());
}

void SteadyClock::sleep_until_ns(std::uint64_t target) {
  target = std::min(target, kMaxDueNs);  // nanoseconds is signed 64-bit
  std::this_thread::sleep_until(origin_ + std::chrono::nanoseconds(static_cast<std::int64_t>(target)));
}

ReplayResult replay(const Model& model, const std::vector<Row>& rows, const ReplayOptions& opts, Clock& clock) {
  if (opts.chunk < 1 || opts.repeat < 1) {
    throw std::invalid_argument("chunk and repeat must be at least 1, got chunk=" + std::to_string(opts.chunk) +
                                ", repeat=" + std::to_string(opts.repeat));
  }
  const std::size_t n = rows.size();
  if (n > 0 && opts.repeat > std::numeric_limits<std::size_t>::max() / n) {
    throw std::invalid_argument("recording too long to repeat that many times");
  }
  const std::size_t total = n * opts.repeat;
  Decoder dec(model);
  ReplayResult r;
  r.samples = total;
  const std::uint64_t start = clock.now_ns();
  for (std::size_t first = 0; first < total; first += opts.chunk) {
    const std::size_t last = std::min(first + opts.chunk, total);
    if (opts.paced) {
      // the clamp only matters past ~292 years of samples, where the cast would be undefined
      const double due_ns = static_cast<double>(last) * 1e9 / model.fs();
      const std::uint64_t due = due_ns >= 9.2e18 ? kMaxDueNs : static_cast<std::uint64_t>(due_ns);
      clock.sleep_until_ns(start + std::min(due, kMaxDueNs - start));
    }
    const std::uint64_t handover = clock.now_ns();
    for (std::size_t j = first; j < last; ++j) {
      const Row& row = rows[j % n];
      if (auto d = dec.push(row.sample, row.stim)) {
        r.latencies_ns.push_back(clock.now_ns() - handover);
        r.decisions.push_back(std::move(*d));
      }
    }
  }
  r.elapsed_ns = clock.now_ns() - start;
  return r;
}

}  // namespace mi
