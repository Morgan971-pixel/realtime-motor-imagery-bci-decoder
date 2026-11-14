#include <gtest/gtest.h>

#include <limits>

#include "mi_core/timing.hpp"

TEST(Timing, DefaultWindowAt250Hz) {
  EXPECT_EQ(mi::to_samples(0.5, 250.0), 125u);
  EXPECT_EQ(mi::to_samples(2.0, 250.0), 500u);
}

TEST(Timing, HalfSamplesRoundUpLikeTheReference) {
  EXPECT_EQ(mi::to_samples(0.5, 125.0), 63u);  // Python's built-in round() would say 62
  EXPECT_EQ(mi::to_samples(0.002, 250.0), 1u);
  EXPECT_EQ(mi::to_samples(0.0, 250.0), 0u);
}

TEST(Timing, HugeProductsSaturateLikeRust) {
  EXPECT_EQ(mi::to_samples(1e300, 250.0), std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(mi::to_samples(1e-300, 250.0), 0u);
}
