// Any bytes into the CSV reader. It must accept or throw mi::CsvError: any other exception, a
// crash or a sanitizer report is a bug. An accepted file has at least one sample.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "mi_core/csv.hpp"
#include "mi_core/error.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  try {
    if (mi::parse_csv(text).empty()) std::abort();
  } catch (const mi::CsvError&) {
    // rejected: the expected outcome for almost every input
  }
  return 0;
}
