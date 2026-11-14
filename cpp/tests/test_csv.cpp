#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "mi_core/csv.hpp"
#include "mi_core/error.hpp"

static const std::string kHead = "timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim\n";

TEST(Csv, ReadsRows) {
  const auto rows = mi::parse_csv(kHead + "1.0,1,2,3,4,5,6,7,8,131\n");
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].sample, (mi::Sample{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(rows[0].stim, 131.0);
  // no trailing newline, tabs around fields
  EXPECT_EQ(mi::parse_csv(kHead + "1\t,\t1,2,3,4,5,6,7,8,0").at(0).sample, (mi::Sample{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(Csv, AcceptsTheDocumentedDialect) {
  const std::string head = "\xEF\xBB\xBFtimestamps, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim\r\n";
  const auto rows = mi::parse_csv(head + "\r\n1, +1, 1.31e2, INF, Infinity, nan, -0.5, 1e400, 7, 131\r\n\n");
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].sample[0], 1.0);
  EXPECT_EQ(rows[0].sample[1], 131.0);
  EXPECT_TRUE(std::isinf(rows[0].sample[2]) && std::isinf(rows[0].sample[3]));
  EXPECT_TRUE(std::isnan(rows[0].sample[4]));
  EXPECT_EQ(rows[0].sample[5], -0.5);
  EXPECT_TRUE(std::isinf(rows[0].sample[6]));  // overflow parses as inf; the decoder rejects it
  EXPECT_EQ(rows[0].sample[7], 7.0);
}

static double one(const std::string& field) {
  return mi::parse_csv(kHead + "0," + field + ",0,0,0,0,0,0,0,0\n").at(0).sample[0];
}

TEST(Csv, OutOfRangeNumbersFollowTheirExponent) {
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(one("1e400"), inf);
  EXPECT_EQ(one("-1e400"), -inf);
  EXPECT_EQ(one("0.001e312"), inf);
  EXPECT_EQ(one("1" + std::string(400, '0')), inf);
  EXPECT_EQ(one("1e99999999999999999999"), inf);
  EXPECT_EQ(one("1e-400"), 0.0);
  EXPECT_FALSE(std::signbit(one("1e-400")));
  EXPECT_TRUE(std::signbit(one("-1e-400")));
  EXPECT_EQ(one("1000e-327"), 0.0);
  EXPECT_EQ(one("0.0000001e-99999999999999999999"), 0.0);
  EXPECT_EQ(one("4.9e-324"), std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(one("1.7976931348623157e308"), std::numeric_limits<double>::max());
}

TEST(Csv, RejectsBadFiles) {
  for (const std::string& bad : {std::string(""), std::string("time,a,b\n1,2,3\n"), kHead, kHead + "1,2,3\n",
                                 kHead + "1,2,3,4,5,6,7,8,9,x\n", kHead + "1,2,3,4,5,6,7,8,9,0x10\n",
                                 kHead + "1,2,3,4,5,6,7,8,9,1_000\n", kHead + "1,2,3,4,5,6,7,8,9,nan(1)\n",
                                 kHead + "# a note\n1,2,3,4,5,6,7,8,9,0\n", kHead + "1,2,3,4,5,6,7,8,9,+-1\n",
                                 std::string("\n"), kHead + "1,2,3,4,5,6,7,8,9,#0\n", kHead + "1,2,3,4,5,6,7,8,9,0,\n",
                                 kHead + "1,2,3,4,5,6,7,8,,0\n", kHead + "1,2,3,4,5,6,7,8,9,.\n",
                                 kHead + "1,2,3,4,5,6,7,8,9,1e\n", kHead + "1,2,3,4,5,6,7,8,9,infx\n"}) {
    EXPECT_THROW(mi::parse_csv(bad), mi::CsvError) << bad;
  }
  try {
    mi::parse_csv(kHead + "1,2,3,4,5,6,7,8,9,0\n\n1,2,3,4,5,6,7,8,9,x\n");
    FAIL() << "accepted a bad stim field";
  } catch (const mi::CsvError& e) {
    EXPECT_STREQ(e.what(), "invalid CSV: line 4, column stim: not a number: \"x\"");
  }
  EXPECT_THROW(mi::read_csv(MI_GOLDEN_DIR), mi::CsvError);  // a directory
  EXPECT_THROW(mi::read_csv(std::string(MI_GOLDEN_DIR) + "/no-such-file.csv"), mi::CsvError);
}
