// Every invalid stream must be rejected at its bad row and leave no trace; boundary codes pass.
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <tuple>

#include "common.hpp"
#include "mi_core/decoder.hpp"
#include "mi_core/error.hpp"

static std::vector<mi::Decision> decisions(mi::Decoder& dec, const std::vector<mi::Row>& rows) {
  std::vector<mi::Decision> out;
  for (const auto& r : rows) {
    if (auto d = dec.push(r.sample, r.stim)) out.push_back(*d);
  }
  return out;
}

/// A valid continuation containing an imagery onset (the golden stream's first trial).
static std::vector<mi::Row> tail() {
  auto rows = mi::read_csv(test::golden("stream.csv"));
  rows.resize(2000);
  return rows;
}

static bool error_at(const std::string& kind, std::size_t bad_row, auto&& call) {
  try {
    call();
  } catch (const mi::StimError& e) {
    return kind == "stim" && e.index() == bad_row;
  } catch (const mi::SampleError& e) {
    return kind == "sample" && e.index() == bad_row;
  }
  return false;
}

TEST(InvalidStreams, StreamingRejectsTheBadRowAndLeavesNoTrace) {
  const auto manifest = test::read_json("invalid/manifest.json");
  bool has_first_row_case = false;
  for (const auto& c : manifest["streams"]) {
    const auto rows = test::manifest_rows(c["rows"]);
    const std::size_t bad = c["bad_row"];
    has_first_row_case |= bad == 0;
    const std::vector<mi::Row> good(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(bad));
    mi::Decoder dec(test::model());
    decisions(dec, good);
    EXPECT_TRUE(error_at(c["kind"], bad, [&] { dec.push(rows[bad].sample, rows[bad].stim); })) << c["name"];
    EXPECT_EQ(dec.samples_seen(), bad) << c["name"];
    mi::Decoder clean(test::model());
    decisions(clean, good);
    const auto after = decisions(dec, tail());
    EXPECT_FALSE(after.empty()) << c["name"];
    EXPECT_EQ(after, decisions(clean, tail())) << c["name"];
  }
  EXPECT_TRUE(has_first_row_case);  // bad_row == 0 catches latching the offset before validating
}

TEST(InvalidStreams, BatchRejectsTheWholeStream) {
  // Named first: a range-for over a member of a temporary dangles in C++20 (ASan: heap-use-after-free).
  const auto manifest = test::read_json("invalid/manifest.json");
  for (const auto& c : manifest["streams"]) {
    const auto rows = test::manifest_rows(c["rows"]);
    EXPECT_TRUE(error_at(c["kind"], c["bad_row"], [&] { mi::decode(test::model(), rows); })) << c["name"];
  }
}

TEST(InvalidStreams, BoundaryCodesAreAccepted) {
  const auto valid = test::read_json("invalid/manifest.json")["valid_streams"];
  ASSERT_FALSE(valid.empty());
  for (const auto& c : valid) {
    const auto rows = test::manifest_rows(c["rows"]);
    EXPECT_NO_THROW(mi::decode(test::model(), rows)) << c["name"];
    mi::Decoder dec(test::model());
    for (const auto& r : rows) dec.push(r.sample, r.stim);
    EXPECT_EQ(dec.samples_seen(), rows.size()) << c["name"];
  }
}

/// Rows run-length encoded as [count, row]; each `rejected` row must fail with its kind and
/// change nothing, so the counters of a run at the rail see only accepted samples.
TEST(InvalidStreams, RejectedRowsInsideARailRunLeaveTheCountersAlone) {
  const auto cases = test::read_json("invalid/manifest.json")["rail_streams"];
  ASSERT_EQ(cases.size(), 3u);
  const auto rail_model = mi::Model::from_path(test::golden("rail_model.json"));
  for (const auto& c : cases) {
    nlohmann::json expanded = nlohmann::json::array();
    for (const auto& run : c["rows"]) {
      for (std::size_t k = 0; k < run[0].get<std::size_t>(); ++k) expanded.push_back(run[1]);
    }
    const auto rows = test::manifest_rows(expanded);
    std::map<std::size_t, std::string> rejected;
    for (const auto& r : c["rejected"]) rejected[r["row"].get<std::size_t>()] = r["kind"].get<std::string>();
    mi::Decoder dec(rail_model);
    mi::Decoder clean(rail_model);
    std::vector<mi::Decision> got, want;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (const auto bad = rejected.find(i); bad != rejected.end()) {
        const std::size_t seen = dec.samples_seen();
        EXPECT_TRUE(error_at(bad->second, seen, [&] { dec.push(rows[i].sample, rows[i].stim); })) << c["name"];
        EXPECT_EQ(dec.samples_seen(), seen) << c["name"];
        continue;
      }
      if (auto d = dec.push(rows[i].sample, rows[i].stim)) got.push_back(*d);
      if (auto d = clean.push(rows[i].sample, rows[i].stim)) want.push_back(*d);
    }
    EXPECT_EQ(got, want) << c["name"];
    using Short = std::tuple<std::size_t, std::uint32_t, std::vector<std::size_t>>;
    std::vector<Short> got_short, expected;
    for (const auto& d : got) got_short.emplace_back(d.sample, d.label, d.pinned);
    for (const auto& e : c["decisions"]) {
      expected.emplace_back(e[0].get<std::size_t>(), e[1].get<std::uint32_t>(), e[2].get<std::vector<std::size_t>>());
    }
    EXPECT_EQ(got_short, expected) << c["name"];
  }
}

TEST(InvalidStreams, EveryValidRailAbsSpellingLoadsAndDecodes) {
  const auto cases = test::read_json("invalid/manifest.json")["valid_models"];
  ASSERT_EQ(cases.size(), 3u);
  const auto stream = mi::read_csv(test::golden("rail_stream.csv"));
  const auto reference = mi::decode(mi::Model::from_path(test::golden("rail_model.json")), stream);
  for (const auto& c : cases) {
    const std::string file = c["file"];
    const auto model = mi::Model::from_path(test::golden("invalid/" + file));
    ASSERT_TRUE(model.rail_abs().has_value()) << file;
    EXPECT_EQ(*model.rail_abs(), c["rail_abs"].get<double>()) << file;
    const auto got = mi::decode(model, stream);
    if (*model.rail_abs() == 1e6) {
      EXPECT_EQ(got, reference) << file;
    } else {
      ASSERT_EQ(got.size(), reference.size()) << file;
      for (const auto& d : got) {
        EXPECT_TRUE(d.pinned.empty()) << file;
        EXPECT_NE(d.label, mi::kRefused) << file;
      }
    }
  }
}
