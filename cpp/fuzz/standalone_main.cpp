// For compilers without libFuzzer (Apple clang): runs LLVMFuzzerTestOneInput once on each file
// named on the command line, so the targets still build and replay seeds or crash inputs.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::ifstream in(argv[i], std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "cannot read %s\n", argv[i]);
      return 1;
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  }
  std::printf("ran %d input(s)\n", argc - 1);
  return 0;
}
