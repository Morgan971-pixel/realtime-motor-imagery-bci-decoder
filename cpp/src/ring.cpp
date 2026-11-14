#include "mi_core/ring.hpp"

#include <stdexcept>

namespace mi {
namespace {

std::size_t checked(std::size_t len) {
  if (len < 2) throw std::invalid_argument("window must span at least 2 samples");
  return len;
}

}  // namespace

Ring::Ring(std::size_t len) : slots_(checked(len), Sample{}) {}

}  // namespace mi
