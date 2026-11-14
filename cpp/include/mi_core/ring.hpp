// Fixed-size window of the last L filtered samples (step 3): sample i goes in slot i % L.
#pragma once

#include <cstddef>
#include <vector>

#include "mi_core/model.hpp"

namespace mi {

class Ring {
 public:
  /// Callers must pass len >= 2; the model loader guarantees it.
  explicit Ring(std::size_t len);
  std::size_t size() const { return slots_.size(); }
  void put(std::size_t i, const Sample& value) { slots_[i % slots_.size()] = value; }
  /// Element k of the window ending at sample i, oldest first (the oldest is slot (i + 1) % L).
  const Sample& oldest_first(std::size_t i, std::size_t k) const { return slots_[(i + 1 + k) % slots_.size()]; }

 private:
  std::vector<Sample> slots_;
};

}  // namespace mi
