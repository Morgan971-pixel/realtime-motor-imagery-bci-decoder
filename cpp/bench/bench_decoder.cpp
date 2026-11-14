// Micro-benchmarks: the per-sample cost of push and a whole golden stream through decode.
// Built by the release preset; run build/release/mi_core_bench.
#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "mi_core/csv.hpp"
#include "mi_core/decoder.hpp"
#include "mi_core/model.hpp"

namespace {

std::string golden(const std::string& name) { return std::string(MI_GOLDEN_DIR) + "/" + name; }

// Markers removed, so no decision ever comes due: this is the cost of filtering and buffering.
void BM_PushSampleWithoutDecision(benchmark::State& state) {
  const mi::Model model = mi::Model::from_path(golden("model.json"));
  std::vector<mi::Row> quiet = mi::read_csv(golden("stream.csv"));
  for (auto& r : quiet) r.stim = 0.0;
  mi::Decoder dec(model);
  for (std::size_t i = 0; i < 1000; ++i) dec.push(quiet[i].sample, quiet[i].stim);
  std::size_t i = 0;
  for (auto _ : state) {
    const mi::Row& r = quiet[i++ % quiet.size()];
    benchmark::DoNotOptimize(dec.push(r.sample, r.stim));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_PushSampleWithoutDecision);

void BM_DecodeGoldenStream(benchmark::State& state) {
  const mi::Model model = mi::Model::from_path(golden("model.json"));
  const std::vector<mi::Row> rows = mi::read_csv(golden("stream.csv"));
  for (auto _ : state) benchmark::DoNotOptimize(mi::decode(model, rows));
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows.size()));
}
BENCHMARK(BM_DecodeGoldenStream)->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
