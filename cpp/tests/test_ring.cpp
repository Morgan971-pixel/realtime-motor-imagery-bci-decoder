#include <gtest/gtest.h>

#include <stdexcept>

#include "mi_core/ring.hpp"

TEST(Ring, WindowIsTheLastLenSamplesOldestFirst) {
  mi::Ring r(3);
  for (std::size_t i = 0; i < 7; ++i) {
    mi::Sample s{};
    s.fill(static_cast<double>(i));
    r.put(i, s);
  }
  EXPECT_EQ(r.oldest_first(6, 0)[0], 4.0);
  EXPECT_EQ(r.oldest_first(6, 1)[0], 5.0);
  EXPECT_EQ(r.oldest_first(6, 2)[0], 6.0);
}

TEST(Ring, EarlyWindowsHoldZerosUntilFull) {
  mi::Ring r(3);
  mi::Sample s{};
  s.fill(1.0);
  r.put(0, s);
  EXPECT_EQ(r.oldest_first(0, 0)[0], 0.0);
  EXPECT_EQ(r.oldest_first(0, 1)[0], 0.0);
  EXPECT_EQ(r.oldest_first(0, 2)[0], 1.0);
  s.fill(2.0);
  r.put(1, s);
  s.fill(3.0);
  r.put(2, s);  // exactly L samples: the window is 1, 2, 3
  EXPECT_EQ(r.oldest_first(2, 0)[0], 1.0);
  EXPECT_EQ(r.oldest_first(2, 2)[0], 3.0);
}

TEST(Ring, RejectsWindowsShorterThanTwo) { EXPECT_THROW(mi::Ring(1), std::invalid_argument); }
