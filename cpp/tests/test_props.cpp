// Randomized tests against independent oracles: the marker rule and the features are recomputed
// here from scratch, without the library's MarkerTracker, Ring or log_variance.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>

#include "common.hpp"
#include "mi_core/decoder.hpp"

namespace {

// 130 and 133 are phase 3 with a movement that is neither left nor right: never a decision
const std::uint32_t kCodes[] = {131, 132, 121, 151, 111, 141, 232, 331, 130, 133};

/// Dense (a code on ~8/9 of samples) or sparse (~1 in 61) stream with a per-channel offset.
std::vector<mi::Row> random_stream(std::mt19937_64& rng, bool dense) {
  std::uniform_int_distribution<std::size_t> length(600, 1599);
  std::uniform_real_distribution<double> offset(-1e4, 1e4), noise(-500.0, 500.0);
  std::bernoulli_distribution has_code(dense ? 8.0 / 9.0 : 1.0 / 61.0);
  std::uniform_int_distribution<std::size_t> pick(0, std::size(kCodes) - 1);
  mi::Sample base;
  for (double& o : base) o = offset(rng);
  std::vector<mi::Row> rows(length(rng));
  for (auto& r : rows) {
    for (std::size_t c = 0; c < mi::kChannels; ++c) r.sample[c] = base[c] + noise(rng);
    r.stim = has_code(rng) ? kCodes[pick(rng)] : 0.0;
  }
  return rows;
}

/// The contract's marker rule, written out independently: (decision sample, trial, movement).
std::vector<std::tuple<std::size_t, std::uint32_t, std::uint32_t>> expected(const std::vector<mi::Row>& rows, std::size_t offset) {
  std::vector<std::tuple<std::size_t, std::uint32_t, std::uint32_t>> out;
  std::uint32_t prev = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto c = static_cast<std::uint32_t>(rows[i].stim);
    const std::uint32_t phase = (c / 10) % 10, movement = c % 10;
    if (c != 0 && c != prev && phase == 3 && (movement == 1 || movement == 2) && i + offset < rows.size()) {
      out.emplace_back(i + offset, c / 100, movement);
    }
    prev = c;
  }
  return out;
}

/// Features recomputed from scratch: filter the whole stream, slice the window, plain loops.
std::vector<double> oracle_features(const mi::Model& model, const std::vector<mi::Row>& rows, std::size_t d) {
  const std::size_t len = model.window_len();
  std::vector<double> features;
  for (const auto& band : model.bands()) {
    mi::SosFilter f(band.sos);
    std::vector<mi::Sample> filtered;
    for (std::size_t i = 0; i <= d; ++i) {
      mi::Sample x = rows[i].sample;
      for (std::size_t c = 0; c < mi::kChannels; ++c) x[c] -= rows[0].sample[c];
      filtered.push_back(f.process(x));
    }
    for (const auto& w : band.csp) {
      std::vector<double> z;
      for (std::size_t t = d + 1 - len; t <= d; ++t) {
        double acc = 0.0;
        for (std::size_t c = 0; c < mi::kChannels; ++c) acc += w[c] * filtered[t][c];
        z.push_back(acc);
      }
      double mean = 0.0;
      for (double v : z) mean += v;
      mean /= static_cast<double>(len);
      double var = 0.0;
      for (double v : z) var += (v - mean) * (v - mean);
      features.push_back(std::log(std::max(var / static_cast<double>(len), 1e-12)));
    }
  }
  return features;
}

}  // namespace

TEST(Props, DecisionsMatchIndependentOraclesOnRandomStreams) {
  const mi::Model& model = test::model();
  const std::size_t offset = model.window_start() + model.window_len() - 1;
  std::mt19937_64 rng(20261107);
  for (int round = 0; round < 48; ++round) {
    const auto rows = random_stream(rng, round % 2 == 0);
    const auto got = mi::decode(model, rows);
    std::vector<std::tuple<std::size_t, std::uint32_t, std::uint32_t>> placed;
    for (const auto& d : got) placed.emplace_back(d.sample, d.trial, d.true_label);
    ASSERT_EQ(placed, expected(rows, offset)) << "round " << round;
    for (const auto& d : got) {
      const auto want = oracle_features(model, rows, d.sample);
      ASSERT_EQ(d.features.size(), want.size()) << "round " << round << " d " << d.sample;
      double s = model.lda_b();
      for (std::size_t j = 0; j < want.size(); ++j) {
        ASSERT_LE(std::abs(d.features[j] - want[j]), std::max(test::kFeatureRel * std::abs(want[j]), 1e-12))
            << "round " << round << " d " << d.sample << " feature " << j;
      }
      double dot = 0.0;
      for (std::size_t j = 0; j < want.size(); ++j) dot += model.lda_w()[j] * want[j];
      s += dot;
      ASSERT_NEAR(d.score, s, test::kScoreAbs) << "round " << round << " d " << d.sample;
      ASSERT_EQ(d.label, d.score > 0.0 ? 2u : 1u);
    }
  }
}

TEST(Props, StreamingEqualsBatchOnRandomStreams) {
  std::mt19937_64 rng(7);
  for (int round = 0; round < 48; ++round) {
    const auto rows = random_stream(rng, round % 2 == 1);
    mi::Decoder dec(test::model());
    std::vector<mi::Decision> streamed;
    for (const auto& r : rows) {
      if (auto d = dec.push(r.sample, r.stim)) streamed.push_back(*d);
    }
    ASSERT_EQ(streamed, mi::decode(test::model(), rows)) << "round " << round;
  }
}
