#include <gtest/gtest.h>

#include <cmath>

#include "mi_core/features.hpp"

TEST(Features, MatchTheExplicitFormula) {
  mi::Ring ring(4);
  const double vals[] = {1, 3, 2, 6};
  for (std::size_t i = 0; i < 4; ++i) ring.put(i, mi::Sample{vals[i]});
  std::vector<double> out;
  mi::log_variance({mi::Sample{1.0}}, ring, 3, out);
  ASSERT_EQ(out.size(), 1u);
  // mean 3, squared deviations 4 + 0 + 1 + 9 = 14, population variance 3.5
  EXPECT_NEAR(out[0], std::log(3.5), 1e-15);
}

TEST(Features, FlatWindowHitsTheFloor) {
  mi::Ring ring(5);
  std::vector<double> out;
  mi::Sample ones;
  ones.fill(1.0);
  mi::log_variance({ones}, ring, 4, out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], std::log(mi::kVarFloor));
}

TEST(Features, NanVarianceTakesTheFloorLikeRust) {
  // an infinite filtered value makes the window variance NaN (inf - inf); the floor must still apply
  mi::Ring ring(3);
  mi::Sample x{};
  x[0] = 1.0 / 0.0;
  ring.put(1, x);
  std::vector<double> out;
  mi::log_variance({mi::Sample{1.0}}, ring, 2, out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], std::log(mi::kVarFloor));
}

TEST(Features, ScoreIsDotPlusBias) { EXPECT_EQ(mi::score({1.0, -2.0}, 0.5, {3.0, 1.0}), 1.5); }
