#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mi_core/stim.hpp"

using mi::Marker;

TEST(Stim, ParseSplitsTrialPhaseMovement) {
  EXPECT_EQ(mi::parse_code(2131), (Marker{21, 3, 1}));
  EXPECT_EQ(mi::parse_code(652), (Marker{6, 5, 2}));
  EXPECT_EQ(mi::parse_code(2147483647u), (Marker{21474836, 4, 7}));
}

TEST(Stim, InvalidValuesAreRejected) {
  for (double bad : {std::nan(""), 1.0 / 0.0, -1.0, 131.5, 31.0, 99.0, 2147483648.0, 2147483647.5}) {
    EXPECT_THROW(mi::validate_code(bad), std::invalid_argument) << bad;
  }
  EXPECT_EQ(mi::validate_code(0.0), 0u);
  EXPECT_EQ(mi::validate_code(-0.0), 0u);
  EXPECT_EQ(mi::validate_code(100.0), 100u);
  EXPECT_EQ(mi::validate_code(2147483647.0), 2147483647u);
}

static std::string reason(double value) {
  try {
    mi::validate_code(value);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "accepted";
}

TEST(Stim, ReasonsAreWordedLikeTheRustCore) {
  EXPECT_EQ(reason(std::nan("")), "NaN or infinite value");
  EXPECT_EQ(reason(-1.0), "negative value -1");
  EXPECT_EQ(reason(131.5), "non-integer value 131.5");
  EXPECT_EQ(reason(2147483648.0), "value 2147483648 above 2147483647");
  EXPECT_EQ(reason(2147483647.5), "value 2147483647.5 above 2147483647");
  EXPECT_EQ(reason(1e300).substr(0, 10), "value 1000");  // plain digits, never 1e+300
  EXPECT_EQ(reason(31.0), "code 31 below 100 (trial >= 1 required)");
}

static std::vector<std::size_t> markers(const std::vector<std::uint32_t>& codes) {
  mi::MarkerTracker t;
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < codes.size(); ++i) {
    if (t.observe(codes[i])) out.push_back(i);
  }
  return out;
}

TEST(Stim, RunsFollowTheContract) {
  EXPECT_EQ(markers({131, 131, 131}), (std::vector<std::size_t>{0}));        // A,A,A: one
  EXPECT_EQ(markers({131, 0, 131}), (std::vector<std::size_t>{0, 2}));       // A,0,A: two
  EXPECT_EQ(markers({131, 151, 131}), (std::vector<std::size_t>{0, 1, 2}));  // A,B,A: three
  EXPECT_EQ(markers({131, 151, 151}), (std::vector<std::size_t>{0, 1}));     // A,B,B: two
  EXPECT_EQ(markers({0, 0, 131}), (std::vector<std::size_t>{2}));
}

TEST(Stim, TrackerReturnsTheParsedMarker) {
  mi::MarkerTracker t;
  EXPECT_EQ(t.observe(2131), (Marker{21, 3, 1}));
  EXPECT_EQ(t.observe(2131), std::nullopt);
  EXPECT_EQ(t.observe(0), std::nullopt);
}

TEST(Stim, OnlyPhase3LeftOrRightIsImagery) {
  EXPECT_TRUE(mi::parse_code(131).is_imagery());
  EXPECT_TRUE(mi::parse_code(232).is_imagery());
  EXPECT_FALSE(mi::parse_code(121).is_imagery());
  EXPECT_FALSE(mi::parse_code(141).is_imagery());
  EXPECT_FALSE(mi::parse_code(133).is_imagery());
}
