// The committed golden files are the contract: the C++ core must reproduce them.
#include <gtest/gtest.h>

#include "common.hpp"
#include "mi_core/decoder.hpp"

/// The model each golden stream is decoded with.
static mi::Model model_for(const std::string& name) {
  return mi::Model::from_path(test::golden(name == "rail_stream" ? "rail_model.json" : "model.json"));
}

static void check_stream(const std::string& name) {
  const mi::Model model = model_for(name);
  const auto expected = test::read_json("expected.json")[name];
  const auto rows = mi::read_csv(test::golden(name + ".csv"));

  // Filtered samples: offset removal (step 1), then each band's filter (step 2).
  for (std::size_t b = 0; b < model.bands().size(); ++b) {
    mi::SosFilter f(model.bands()[b].sos);
    std::vector<mi::Sample> out;
    for (const auto& r : rows) {
      mi::Sample x = r.sample;
      for (std::size_t c = 0; c < mi::kChannels; ++c) x[c] -= rows[0].sample[c];
      out.push_back(f.process(x));
    }
    const auto& tail = expected["filtered_tail_per_band"][b];
    for (std::size_t k = 0; k < tail.size(); ++k) {
      for (std::size_t c = 0; c < mi::kChannels; ++c) {
        ASSERT_NEAR(out[out.size() - tail.size() + k][c], tail[k][c].get<double>(), test::kFilterAbs) << name << " band " << b;
      }
    }
    if (b == 0) {
      const auto& head = expected["band0_filtered_head"];
      for (std::size_t k = 0; k < head.size(); ++k) {
        for (std::size_t c = 0; c < mi::kChannels; ++c) {
          ASSERT_NEAR(out[k][c], head[k][c].get<double>(), test::kFilterAbs) << name << " head";
        }
      }
    }
  }

  // Decisions: exact sample, trial, labels and count; features rel 1e-9; score abs 1e-7.
  const auto got = mi::decode(model, rows);
  const auto& want = expected["decisions"];
  ASSERT_EQ(got.size(), want.size()) << name;
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].sample, want[i]["sample"].get<std::size_t>());
    EXPECT_EQ(got[i].trial, want[i]["trial"].get<std::uint32_t>());
    EXPECT_EQ(got[i].true_label, want[i]["true_label"].get<std::uint32_t>());
    EXPECT_EQ(got[i].label, want[i]["label"].get<std::uint32_t>());
    EXPECT_EQ(got[i].pinned, want[i]["pinned"].get<std::vector<std::size_t>>()) << name << " " << got[i].sample;
    ASSERT_EQ(got[i].features.size(), want[i]["features"].size());
    for (std::size_t j = 0; j < got[i].features.size(); ++j) {
      EXPECT_TRUE(test::close_rel(got[i].features[j], want[i]["features"][j].get<double>(), test::kFeatureRel));
    }
    EXPECT_NEAR(got[i].score, want[i]["score"].get<double>(), test::kScoreAbs);
  }
}

TEST(Golden, ReproducesTheGoldenStream) { check_stream("stream"); }
TEST(Golden, ReproducesTheGoldenEdgeStream) { check_stream("edge_stream"); }
TEST(Golden, ReproducesTheGoldenRailStream) { check_stream("rail_stream"); }

TEST(Golden, StreamingPushReproducesBatchOnEveryStream) {
  for (const std::string name : {"stream", "edge_stream", "rail_stream"}) {
    const auto rows = mi::read_csv(test::golden(name + ".csv"));
    mi::Decoder dec(model_for(name));
    std::vector<mi::Decision> streamed;
    for (const auto& r : rows) {
      if (auto d = dec.push(r.sample, r.stim)) streamed.push_back(*d);
    }
    EXPECT_EQ(streamed, mi::decode(model_for(name), rows)) << name;
  }
}
