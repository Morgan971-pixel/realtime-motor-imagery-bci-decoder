#include "mi_core/decoder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "mi_core/error.hpp"
#include "mi_core/features.hpp"

namespace mi {

std::uint32_t check_row(std::size_t index, const Sample& sample, double stim) {
  for (double v : sample) {
    if (!std::isfinite(v)) throw SampleError(index, "NaN or infinite value");
  }
  try {
    return validate_code(stim);
  } catch (const std::invalid_argument& e) {
    throw StimError(index, e.what());
  }
}

Decoder::Decoder(Model model)
    : model_(std::move(model)), start_(model_.window_start()), len_(model_.window_len()) {
  for (const Band& band : model_.bands()) {
    filters_.emplace_back(band.sos);
    rings_.emplace_back(len_);
  }
}

std::optional<Decision> Decoder::push(const Sample& sample, double stim) {
  const std::size_t i = next_;
  const std::uint32_t code = check_row(i, sample, stim);

  ++next_;
  if (const auto rail = model_.rail_abs()) {  // raw values, before the offset is removed
    for (std::size_t c = 0; c < kChannels; ++c) {
      const bool at_rail = std::abs(sample[c]) >= *rail;
      run_[c] = at_rail ? std::min(run_[c] + 1, len_) : 0;  // saturates at L (at most 2^20): pinned needs run >= L
    }
  }
  if (!x0_) x0_ = sample;
  Sample x{};
  for (std::size_t c = 0; c < kChannels; ++c) x[c] = sample[c] - (*x0_)[c];
  for (std::size_t b = 0; b < filters_.size(); ++b) rings_[b].put(i, filters_[b].process(x));
  if (const auto m = markers_.observe(code); m && m->is_imagery()) {
    // start and len are each at most 2^20, so this only overflows after ~2^64 samples on 64-bit
    pending_.push_back(Pending{i + start_ + len_ - 1, m->trial, m->movement});
  }
  assert((pending_.empty() || pending_.front().d >= i) && "missed a decision");
  if (pending_.empty() || pending_.front().d != i) return std::nullopt;
  Decision d = decide(i, pending_.front().trial, pending_.front().movement);  // may allocate: pop only after
  pending_.pop_front();
  return d;
}

Decision Decoder::decide(std::size_t i, std::uint32_t trial, std::uint32_t movement) const {
  std::vector<double> features;
  features.reserve(model_.lda_w().size());
  for (std::size_t b = 0; b < rings_.size(); ++b) log_variance(model_.bands()[b].csp, rings_[b], i, features);
  const double s = score(model_.lda_w(), model_.lda_b(), features);
  std::vector<std::size_t> pinned;
  if (model_.rail_abs()) {
    for (std::size_t c : model_.used_channels()) {
      if (run_[c] >= len_) pinned.push_back(c);
    }
  }
  const std::uint32_t label = !pinned.empty() ? kRefused : s > 0.0 ? kRight : kLeft;
  return Decision{i, trial, movement, std::move(features), s, label, std::move(pinned)};
}

std::vector<Decision> decode(const Model& model, const std::vector<Row>& rows) {
  for (std::size_t i = 0; i < rows.size(); ++i) check_row(i, rows[i].sample, rows[i].stim);
  Decoder dec(model);
  std::vector<Decision> out;
  for (const Row& row : rows) {
    if (auto d = dec.push(row.sample, row.stim)) out.push_back(std::move(*d));
  }
  return out;
}

}  // namespace mi
