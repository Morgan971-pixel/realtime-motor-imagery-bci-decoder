// Club CSV reader: header timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim, one row per sample.
// Values are parsed as plain doubles here (including nan and inf); the decoder enforces the
// stim and EEG rules, so a bad value is reported with its sample index.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "mi_core/model.hpp"

namespace mi {

struct Row {
  Sample sample;
  double stim;
  bool operator==(const Row&) const = default;
};

std::vector<Row> parse_csv(std::string_view text);
std::vector<Row> read_csv(const std::string& path);

}  // namespace mi
