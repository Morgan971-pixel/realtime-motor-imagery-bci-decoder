// Strict model.json loader: every load error in the contract throws ModelError, never a fallback.
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mi {

constexpr std::size_t kChannels = 8;
/// Window start and length may each span at most 2^20 samples (about 70 min at 250 Hz).
constexpr double kMaxWindowSamples = 1048576.0;

using Sample = std::array<double, kChannels>;

struct Band {
  std::vector<std::array<double, 6>> sos;  ///< sections [b0, b1, b2, a0, a1, a2], in order
  std::vector<Sample> csp;                 ///< CSP rows, one weight per channel
};

/// A validated model. The constructor is private: from_json / from_path are the only way in.
class Model {
 public:
  static Model from_json(std::string_view text);
  static Model from_path(const std::string& path);

  double fs() const { return fs_; }
  const std::vector<Band>& bands() const { return bands_; }
  const std::vector<double>& lda_w() const { return lda_w_; }
  double lda_b() const { return lda_b_; }
  /// Raw value at or beyond which a channel counts as pinned; empty turns the check off.
  std::optional<double> rail_abs() const { return rail_abs_; }
  /// Channels with a nonzero CSP weight in some band, ascending.
  const std::vector<std::size_t>& used_channels() const { return used_channels_; }
  std::size_t window_len() const;    ///< L in samples
  std::size_t window_start() const;  ///< start offset after the onset, in samples

 private:
  Model() = default;
  double fs_ = 0.0;
  double window_start_s_ = 0.0;
  double window_len_s_ = 0.0;
  std::vector<Band> bands_;
  std::vector<double> lda_w_;
  double lda_b_ = 0.0;
  std::optional<double> rail_abs_;
  std::vector<std::size_t> used_channels_;
};

}  // namespace mi
