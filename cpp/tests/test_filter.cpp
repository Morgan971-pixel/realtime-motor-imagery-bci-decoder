#include <gtest/gtest.h>

#include <vector>

#include "mi_core/filter.hpp"

static std::vector<double> impulse(std::vector<std::array<double, 6>> sections, std::size_t n) {
  mi::SosFilter f(std::move(sections));
  std::vector<double> out;
  for (std::size_t k = 0; k < n; ++k) {
    mi::Sample x{};
    x.fill(k == 0 ? 1.0 : 0.0);
    out.push_back(f.process(x)[0]);
  }
  return out;
}

TEST(Filter, IdentitySectionPassesInputThrough) {
  mi::SosFilter f({{1, 0, 0, 1, 0, 0}});
  const mi::Sample x{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(f.process(x), x);
}

TEST(Filter, FilterIsCausalAndKeepsState) {
  // y[n] = x[n] + x[n-1] (b0 = b1 = 1): the state carries the previous input between calls
  mi::SosFilter f({{1, 1, 0, 1, 0, 0}});
  mi::Sample one;
  one.fill(1.0);
  EXPECT_EQ(f.process(mi::Sample{})[0], 0.0);
  EXPECT_EQ(f.process(one)[0], 1.0);
  EXPECT_EQ(f.process(mi::Sample{})[0], 1.0);
  EXPECT_EQ(f.process(mi::Sample{})[0], 0.0);
}

TEST(Filter, ChannelsKeepSeparateState) {
  // channel c gets its impulse at time c, so it must see the single-channel response shifted by c
  const std::vector<std::array<double, 6>> sections{{1, 0, 0, 1, -0.5, 0}, {0, 1, 0, 1, 0, -0.25}};
  const std::size_t n = 16;
  const auto h = impulse(sections, n);
  mi::SosFilter f(sections);
  for (std::size_t k = 0; k < n; ++k) {
    mi::Sample x{};
    if (k < x.size()) x[k] = 1.0;
    const mi::Sample y = f.process(x);
    for (std::size_t c = 0; c < y.size(); ++c) EXPECT_EQ(y[c], k >= c ? h[k - c] : 0.0) << "k=" << k << " c=" << c;
  }
}

TEST(Filter, FeedbackTermsHaveTheContractSigns) {
  // powers of two keep these comparisons exact
  EXPECT_EQ(impulse({{1, 0, 0, 1, -0.5, 0}}, 4), (std::vector<double>{1, 0.5, 0.25, 0.125}));
  EXPECT_EQ(impulse({{1, 0, 0, 1, 0, -0.25}}, 5), (std::vector<double>{1, 0, 0.25, 0, 0.0625}));
}

TEST(Filter, B2DelaysByTwoSamplesAndSectionsCascade) {
  EXPECT_EQ(impulse({{0, 0, 1, 1, 0, 0}}, 4), (std::vector<double>{0, 0, 1, 0}));
  EXPECT_EQ(impulse({{0, 1, 0, 1, 0, 0}, {0, 0, 1, 1, 0, 0}}, 5), (std::vector<double>{0, 0, 0, 1, 0}));
}

TEST(Filter, ZeroInputGivesExactlyZeroOutput) {
  mi::SosFilter f(std::vector<std::array<double, 6>>(4, {0.3, 0.1, -0.2, 1.0, -0.5, 0.25}));
  for (int k = 0; k < 100; ++k) EXPECT_EQ(f.process(mi::Sample{}), mi::Sample{});
}
