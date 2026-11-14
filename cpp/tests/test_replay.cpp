// Replay (docs/replay.md): the same decisions as batch decoding, timed per chunk.
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "mi_core/decoder.hpp"
#include "mi_core/replay.hpp"

namespace {

/// Every now_ns() call advances time by 1000 ns; a sleep wakes 0.5 ms late, as on macOS.
class FakeClock final : public mi::Clock {
 public:
  std::uint64_t now_ns() override { return t += 1000; }
  void sleep_until_ns(std::uint64_t target) override {
    sleeps.push_back(target);
    t = std::max(t, target + 500'000);
  }
  std::uint64_t t = 0;
  std::vector<std::uint64_t> sleeps;
};

const std::vector<mi::Row>& edge_rows() {
  static const auto rows = mi::read_csv(test::golden("edge_stream.csv"));
  return rows;
}

}  // namespace

TEST(Replay, MakesTheBatchDecisionsForAnyChunkSize) {
  const auto want = mi::decode(test::model(), edge_rows());
  for (std::size_t chunk : {1u, 7u, 25u}) {
    mi::SteadyClock clock;
    const auto r = mi::replay(test::model(), edge_rows(), {.chunk = chunk, .paced = false, .repeat = 1}, clock);
    EXPECT_EQ(r.decisions, want) << "chunk " << chunk;
    EXPECT_EQ(r.samples, edge_rows().size());
    ASSERT_EQ(r.latencies_ns.size(), r.decisions.size());
    for (auto x : r.latencies_ns) EXPECT_LE(x, r.elapsed_ns);
  }
}

TEST(Replay, RepeatPlaysTheRecordingBackToBack) {
  std::vector<mi::Row> twice = edge_rows();
  twice.insert(twice.end(), edge_rows().begin(), edge_rows().end());
  // 9250 % 7 = 3: with chunk 7 a chunk straddles the wrap between passes
  for (std::size_t chunk : {7u, 25u}) {
    mi::SteadyClock clock;
    const auto r = mi::replay(test::model(), edge_rows(), {.chunk = chunk, .paced = false, .repeat = 2}, clock);
    EXPECT_EQ(r.samples, twice.size());
    EXPECT_EQ(r.decisions, mi::decode(test::model(), twice)) << "chunk " << chunk;
  }
}

TEST(Replay, FastModeTimesEachDecisionFromItsChunkHandover) {
  FakeClock clock;
  const auto r = mi::replay(test::model(), edge_rows(), {.chunk = 25, .paced = false, .repeat = 1}, clock);
  EXPECT_TRUE(clock.sleeps.empty());
  EXPECT_EQ(r.latencies_ns, std::vector<std::uint64_t>(9, 1000));
}

TEST(Replay, PacedModeWaitsForEachChunkThenTimesFromItsHandover) {
  const std::size_t total = 2 * edge_rows().size();
  FakeClock clock;
  const auto r = mi::replay(test::model(), edge_rows(), {.chunk = 7, .paced = true, .repeat = 2}, clock);
  // chunk k is due when its last sample has arrived: start + (samples so far) / fs; the last
  // chunk is short (18500 % 7 = 6) and one chunk spans the wrap between the two passes
  const auto due_at = [](std::size_t end) { return 1000 + static_cast<std::uint64_t>(static_cast<double>(end) * 1e9 / test::model().fs()); };
  std::vector<std::uint64_t> due;
  for (std::size_t end = 7; end < total; end += 7) due.push_back(due_at(end));
  due.push_back(due_at(total));
  EXPECT_EQ(clock.sleeps, due);
  // one clock reading per decision since the handover: a second decision in the same chunk
  // (samples 2724 and 2726 share one) reads 2000 ns
  std::vector<std::uint64_t> want;
  for (std::size_t i = 0; i < r.decisions.size(); ++i) {
    std::uint64_t k = 0;
    for (std::size_t j = 0; j <= i; ++j) k += r.decisions[j].sample / 7 == r.decisions[i].sample / 7;
    want.push_back(1000 * k);
  }
  ASSERT_EQ(r.decisions.size(), 20u);
  EXPECT_NE(std::find(want.begin(), want.end(), 2000u), want.end());
  EXPECT_EQ(r.latencies_ns, want);
  EXPECT_GE(r.elapsed_ns, due_at(total) - 1000);
}

TEST(Replay, PacedModeReallyRunsAtTheSamplingRate) {
  const std::vector<mi::Row> rows(edge_rows().begin(), edge_rows().begin() + 50);
  mi::SteadyClock clock;
  const auto r = mi::replay(test::model(), rows, {.chunk = 25, .paced = true, .repeat = 1}, clock);
  EXPECT_GE(r.elapsed_ns, 200'000'000u) << "50 samples at 250 Hz take at least 0.2 s";
}

TEST(Replay, ChunkAndRepeatMustBePositive) {
  mi::SteadyClock clock;
  EXPECT_THROW(mi::replay(test::model(), edge_rows(), {.chunk = 0, .paced = false, .repeat = 1}, clock), std::invalid_argument);
  EXPECT_THROW(mi::replay(test::model(), edge_rows(), {.chunk = 25, .paced = false, .repeat = 0}, clock), std::invalid_argument);
}

TEST(Replay, RepeatThatOverflowsTheSampleCountIsRejected) {
  mi::SteadyClock clock;
  const std::size_t repeat = std::numeric_limits<std::size_t>::max() / edge_rows().size() + 1;
  EXPECT_THROW(mi::replay(test::model(), edge_rows(), {.chunk = 25, .paced = false, .repeat = repeat}, clock), std::invalid_argument);
}
