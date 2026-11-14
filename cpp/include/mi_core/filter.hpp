// One band's causal filter: second-order sections in transposed direct form II (step 2).
#pragma once

#include <array>
#include <vector>

#include "mi_core/model.hpp"

namespace mi {

class SosFilter {
 public:
  explicit SosFilter(std::vector<std::array<double, 6>> sections);
  /// Filter one multichannel sample. Same operations, same order, as the contract.
  Sample process(const Sample& x);

 private:
  std::vector<std::array<double, 6>> sections_;
  std::vector<std::array<std::array<double, 2>, kChannels>> state_;  ///< per section, per channel: z0, z1
};

}  // namespace mi
