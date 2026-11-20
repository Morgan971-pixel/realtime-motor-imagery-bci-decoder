#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "common.hpp"
#include "mi_core/error.hpp"
#include "mi_core/json_strict.hpp"

TEST(Model, GoldenModelLoadsWithTheExpectedShape) {
  const mi::Model& m = test::model();
  EXPECT_EQ(m.fs(), 250.0);
  EXPECT_EQ(m.window_start(), 125u);
  EXPECT_EQ(m.window_len(), 500u);
  ASSERT_EQ(m.bands().size(), 4u);
  for (const auto& b : m.bands()) {
    EXPECT_EQ(b.sos.size(), 4u);
    EXPECT_EQ(b.csp.size(), 4u);
  }
  EXPECT_EQ(m.lda_w().size(), 16u);
}

TEST(Model, EveryInvalidModelIsRejectedForItsTaggedRule) {
  const auto manifest = test::read_json("invalid/manifest.json");
  // every invalid model on disk is listed (valid_* files are the valid spellings), and only those
  std::vector<std::string> on_disk, listed;
  for (const auto& entry : std::filesystem::directory_iterator(test::golden("invalid"))) {
    const std::string name = entry.path().filename().string();
    if (entry.path().extension() == ".json" && name != "manifest.json" && name.rfind("valid_", 0) != 0) {
      on_disk.push_back(name);
    }
  }
  for (const auto& c : manifest["models"]) listed.push_back(c["file"]);
  std::sort(on_disk.begin(), on_disk.end());
  std::sort(listed.begin(), listed.end());
  ASSERT_EQ(listed, on_disk);
  for (const auto& c : manifest["models"]) {
    const std::string file = c["file"];
    try {
      mi::Model::from_json(test::read_text(test::golden("invalid/" + file)));
      ADD_FAILURE() << file << " (" << c["reason"].get<std::string>() << ") was not rejected";
    } catch (const mi::ModelError& e) {
      EXPECT_EQ(e.rule(), c["rule"].get<std::string>()) << file << ": " << e.what();
    }
  }
}

// Found by the Hypothesis property test_native_core_reads_any_spelling_of_the_same_numbers: the
// integer spelling -0 lost its sign in C++ only, so Rust and C++ scores differed in the sign of 0.
TEST(Model, NegativeZeroWrittenAsAnIntegerKeepsItsSign) {
  const auto doc = mi::parse_strict("[-0, 0, -0.0, -1]");
  EXPECT_TRUE(std::signbit(doc[0].get<double>()));
  EXPECT_EQ(doc[0].get<double>(), 0.0);
  EXPECT_FALSE(std::signbit(doc[1].get<double>()));
  EXPECT_TRUE(std::signbit(doc[2].get<double>()));
  EXPECT_EQ(doc[3].get<double>(), -1.0);
}

