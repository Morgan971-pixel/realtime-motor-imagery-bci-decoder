// The streaming decoder (steps 1-6) and the batch helper built on it.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "mi_core/csv.hpp"
#include "mi_core/filter.hpp"
#include "mi_core/model.hpp"
#include "mi_core/ring.hpp"
#include "mi_core/stim.hpp"

namespace mi {

/// Label of a refused decision: a used channel sat at or beyond rail_abs for the whole window.
constexpr std::uint32_t kRefused = 0;

struct Decision {
  std::size_t sample;         ///< sample index d at which the decision is made
  std::uint32_t trial;
  std::uint32_t true_label;   ///< the marker's movement (1 = left, 2 = right)
  std::vector<double> features;
  double score;
  std::uint32_t label;        ///< REFUSED (0) if any channel is pinned, else RIGHT (2) if score > 0, else LEFT (1)
  std::vector<std::size_t> pinned;  ///< used channels at or beyond rail_abs all window, ascending
  bool operator==(const Decision&) const = default;
};

class Decoder {
 public:
  explicit Decoder(Model model);
  /// Feed one sample. Invalid input throws SampleError or StimError before any state changes.
  std::optional<Decision> push(const Sample& sample, double stim);
  std::size_t samples_seen() const { return next_; }

 private:
  friend struct DecoderTestPeer;  ///< the unit tests read run_ (tests/test_pinned.cpp)

  Decision decide(std::size_t i, std::uint32_t trial, std::uint32_t movement) const;

  struct Pending {
    std::size_t d;
    std::uint32_t trial;
    std::uint32_t movement;
  };
  Model model_;
  std::vector<SosFilter> filters_;
  std::vector<Ring> rings_;
  std::optional<Sample> x0_;
  std::size_t next_ = 0;
  MarkerTracker markers_;
  std::deque<Pending> pending_;  ///< d only grows, so a queue suffices
  std::size_t start_;
  std::size_t len_;
  std::array<std::size_t, kChannels> run_{};  ///< per channel: consecutive samples at or beyond rail_abs, at most L
};

/// The contract's input checks for one row, EEG first, then stim. Returns the stim code.
/// The single source of these rules for both streaming and batch decoding.
std::uint32_t check_row(std::size_t index, const Sample& sample, double stim);

/// Batch decoding: validates every row first (rejecting the whole stream), then streams.
std::vector<Decision> decode(const Model& model, const std::vector<Row>& rows);

}  // namespace mi
