// Any bytes into the model loader. It must accept or throw mi::ModelError: any other exception,
// a crash or a sanitizer report is a bug. An accepted model keeps the contract's window limits, a
// rail_abs (when present) that is finite and > 0, and used channels in strictly ascending order.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "mi_core/error.hpp"
#include "mi_core/model.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  constexpr std::size_t kMaxWindow = std::size_t{1} << 20;
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  try {
    const mi::Model m = mi::Model::from_json(text);
    if (m.window_len() < 2 || m.window_len() > kMaxWindow || m.window_start() > kMaxWindow || m.bands().empty()) {
      std::abort();
    }
    if (m.rail_abs() && !(std::isfinite(*m.rail_abs()) && *m.rail_abs() > 0.0)) std::abort();
    const auto& used = m.used_channels();
    for (std::size_t i = 0; i < used.size(); ++i) {
      if (used[i] >= 8 || (i > 0 && used[i - 1] >= used[i])) std::abort();
    }
  } catch (const mi::ModelError&) {
    // rejected: the expected outcome for almost every input
  }
  return 0;
}
