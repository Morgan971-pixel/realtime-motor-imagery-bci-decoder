#include "mi_core/filter.hpp"

#include <utility>

namespace mi {

SosFilter::SosFilter(std::vector<std::array<double, 6>> sections)
    : sections_(std::move(sections)), state_(sections_.size()) {
  for (auto& section_state : state_) {
    for (auto& z : section_state) z = {0.0, 0.0};
  }
}

Sample SosFilter::process(const Sample& x) {
  Sample cur = x;
  for (std::size_t s = 0; s < sections_.size(); ++s) {
    const auto& [b0, b1, b2, a0, a1, a2] = sections_[s];
    (void)a0;
    for (std::size_t c = 0; c < kChannels; ++c) {
      auto& z = state_[s][c];
      const double out = b0 * cur[c] + z[0];
      z[0] = b1 * cur[c] - a1 * out + z[1];
      z[1] = b2 * cur[c] - a2 * out;
      cur[c] = out;
    }
  }
  return cur;
}

}  // namespace mi
