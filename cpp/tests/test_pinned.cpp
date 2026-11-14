// The pinned-channel check (rail_abs): the same cases as python/tests/test_pinned.py.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "common.hpp"
#include "mi_core/decoder.hpp"
#include "mi_core/error.hpp"

namespace {

constexpr std::size_t kLen = 500;
constexpr std::size_t kOnset = 10;
constexpr std::size_t kD = kOnset + 125 + kLen - 1;  // the one decision sample: 634
constexpr std::size_t kN = kD + 5;
constexpr double kRail = 1e6;

/// The golden model with rail_abs (empty leaves it out) and no CSP weight on `unused`.
mi::Model model(std::optional<double> rail, const std::vector<std::size_t>& unused = {}) {
  auto j = test::read_json("model.json");
  if (rail) j["rail_abs"] = *rail;
  for (auto& band : j["bands"]) {
    for (auto& row : band["csp"]) {
      for (std::size_t c : unused) row[c] = 0.0;
    }
  }
  return mi::Model::from_json(j.dump());
}

/// A quiet stream with one imagery onset; `edit` drives channels to the rail.
mi::Decision run(const mi::Model& m, const std::function<void(std::size_t, mi::Sample&)>& edit) {
  mi::Decoder dec(m);
  std::vector<mi::Decision> out;
  for (std::size_t i = 0; i < kN; ++i) {
    mi::Sample s;
    s.fill(5e5 + static_cast<double>(i % 7));
    edit(i, s);
    if (auto d = dec.push(s, i == kOnset ? 131.0 : 0.0)) out.push_back(*d);
  }
  EXPECT_EQ(out.size(), 1u);
  EXPECT_EQ(out.at(0).sample, kD);
  return out.at(0);
}

bool in_window(std::size_t i, std::size_t first) { return i >= first && i <= kD; }

std::string load_error_text(const std::string& text) {
  try {
    mi::Model::from_json(text);
  } catch (const mi::ModelError& e) {
    return e.rule() + ": " + e.what();
  }
  return "accepted";
}

std::string load_error(const nlohmann::json& value) {
  auto j = test::read_json("model.json");
  j["rail_abs"] = value;
  return load_error_text(j.dump());
}

/// The golden model's text with `entries` spliced in before "labels" (raw text, so a key can
/// repeat and a number can overflow).
std::string with_raw(const std::string& entries) {
  std::string text = test::read_text(test::golden("model.json"));
  const auto at = text.find("\"labels\": ");
  EXPECT_NE(at, std::string::npos);
  return text.insert(at, entries);
}

}  // namespace

namespace mi {
struct DecoderTestPeer {
  static const std::array<std::size_t, kChannels>& run(const Decoder& d) { return d.run_; }
};
}  // namespace mi

TEST(Pinned, WithoutRailAbsNothingIsEverRefused) {
  const auto d = run(model(std::nullopt), [](std::size_t, mi::Sample& s) { s[2] = kRail; });
  EXPECT_TRUE(d.pinned.empty());
  EXPECT_NE(d.label, mi::kRefused);
}

TEST(Pinned, AUsedChannelAtTheRailForExactlyTheWindowRefuses) {
  const auto d = run(model(kRail), [](std::size_t i, mi::Sample& s) {
    if (in_window(i, kD + 1 - kLen)) s[2] = kRail;
  });
  EXPECT_EQ(d.label, mi::kRefused);
  EXPECT_EQ(d.pinned, std::vector<std::size_t>{2});
  EXPECT_TRUE(std::isfinite(d.score));
  EXPECT_EQ(d.features.size(), 16u);
}

TEST(Pinned, OneSampleShortOfTheWindowDecides) {
  const auto d = run(model(kRail), [](std::size_t i, mi::Sample& s) {
    if (in_window(i, kD + 2 - kLen)) s[2] = kRail;
  });
  EXPECT_TRUE(d.pinned.empty());
  EXPECT_NE(d.label, mi::kRefused);
}

TEST(Pinned, BothSignsCountAndExactlyRailAbsIsPinned) {
  const auto d = run(model(kRail), [](std::size_t, mi::Sample& s) {
    s[5] = -kRail;
    s[3] = kRail * (1.0 - 1e-12);  // just inside: never pinned
  });
  EXPECT_EQ(d.pinned, std::vector<std::size_t>{5});
}

TEST(Pinned, AnUnusedChannelNeverRefuses) {
  const auto m = model(kRail, {6});
  EXPECT_EQ(m.used_channels(), (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 7}));
  const auto d = run(m, [](std::size_t, mi::Sample& s) { s[6] = kRail; });
  EXPECT_TRUE(d.pinned.empty());
}

TEST(Pinned, SeveralPinnedChannelsComeOutInChannelOrder) {
  const auto d = run(model(kRail), [](std::size_t, mi::Sample& s) {
    s[7] = kRail;
    s[0] = kRail;
    s[4] = -2.0 * kRail;
  });
  EXPECT_EQ(d.pinned, (std::vector<std::size_t>{0, 4, 7}));
  EXPECT_EQ(d.label, mi::kRefused);
}

TEST(Pinned, ARejectedSampleDoesNotTouchTheRunCounters) {
  // L - 1 accepted samples at the rail, plus two rejected rows at the rail inside the run (one
  // with NaN on another channel, one with a bad stim): counting either would make L
  mi::Decoder dec(model(kRail));
  std::vector<mi::Decision> out;
  for (std::size_t i = 0; i < kN; ++i) {
    mi::Sample s;
    s.fill(5e5);
    if (i > kD + 1 - kLen) s[2] = kRail;
    if (i == kD - 10) {
      mi::Sample nan = s;
      nan[0] = std::numeric_limits<double>::quiet_NaN();
      EXPECT_THROW(dec.push(nan, 0.0), mi::SampleError);
      EXPECT_THROW(dec.push(s, 99.0), mi::StimError);
    }
    if (auto d = dec.push(s, i == kOnset ? 131.0 : 0.0)) out.push_back(*d);
  }
  ASSERT_EQ(out.size(), 1u);
  EXPECT_TRUE(out[0].pinned.empty());
  EXPECT_NE(out[0].label, mi::kRefused);
}

TEST(Pinned, ADipInTheMiddleOfTheWindowResetsTheRun) {
  const auto d = run(model(kRail), [](std::size_t i, mi::Sample& s) {
    if (i != kD - kLen / 2) s[3] = -kRail;  // the rail counts both signs
  });
  EXPECT_TRUE(d.pinned.empty());
  EXPECT_NE(d.label, mi::kRefused);
}

TEST(Pinned, RailAbsIsOptionalAndAnyJsonNumber) {
  EXPECT_FALSE(model(std::nullopt).rail_abs().has_value());
  auto j = test::read_json("model.json");
  j["rail_abs"] = 700000;
  EXPECT_EQ(mi::Model::from_json(j.dump()).rail_abs(), 700000.0);
}

TEST(Pinned, BadRailAbsIsALoadErrorWithItsRule) {
  EXPECT_EQ(load_error(0.0).rfind("value: ", 0), 0u) << load_error(0.0);
  EXPECT_EQ(load_error(-1.0).rfind("value: ", 0), 0u);
  EXPECT_EQ(load_error("1e6").rfind("type: ", 0), 0u);
  EXPECT_EQ(load_error(nullptr).rfind("type: ", 0), 0u);
  EXPECT_EQ(load_error(true).rfind("type: ", 0), 0u);
  EXPECT_NE(load_error(0.0).find("rail_abs must be finite and > 0"), std::string::npos);
  EXPECT_EQ(load_error_text(with_raw("\"rail_abs\": 1e400, ")).rfind("json: ", 0), 0u);
  EXPECT_EQ(load_error_text(with_raw("\"rail_abs\": 1" + std::string(400, '0') + ", ")).rfind("json: ", 0), 0u);
  EXPECT_EQ(load_error_text(with_raw("\"rail_abs\": 1e6, \"rail_abs\": 1e6, ")).rfind("keys: ", 0), 0u);
  EXPECT_EQ(load_error_text(with_raw("\"rail_abs\": 1e6, ")), "accepted");  // the splice itself is valid
}

TEST(Pinned, RunCountersSaturateAtTheWindowLength) {
  mi::Decoder dec(model(kRail));
  mi::Sample s;
  s.fill(5e5);
  s[2] = kRail;
  for (std::size_t i = 0; i < 2 * kLen + 7; ++i) dec.push(s, 0.0);
  EXPECT_EQ(mi::DecoderTestPeer::run(dec), (std::array<std::size_t, mi::kChannels>{0, 0, kLen, 0, 0, 0, 0, 0}));
  s[2] = 5e5;
  dec.push(s, 0.0);
  EXPECT_EQ(mi::DecoderTestPeer::run(dec), (std::array<std::size_t, mi::kChannels>{}));
}
