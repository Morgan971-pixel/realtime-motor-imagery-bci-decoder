#include "mi_core/features.hpp"

#include <cmath>

namespace mi {

void log_variance(const std::vector<Sample>& csp, const Ring& ring, std::size_t i, std::vector<double>& out) {
  const std::size_t n = ring.size();
  const double len = static_cast<double>(n);
  std::vector<double> z(n);
  for (const Sample& w : csp) {
    for (std::size_t k = 0; k < n; ++k) {
      const Sample& x = ring.oldest_first(i, k);
      double acc = 0.0;
      for (std::size_t c = 0; c < kChannels; ++c) acc += w[c] * x[c];
      z[k] = acc;
    }
    double sum = -0.0;  // Rust's f64 Sum starts at -0.0; matching it keeps even zero signs identical
    for (double v : z) sum += v;
    const double mean = sum / len;
    double ss = -0.0;
    for (double v : z) ss += (v - mean) * (v - mean);
    out.push_back(std::log(std::fmax(ss / len, kVarFloor)));  // fmax: a NaN variance takes the floor, like Rust's f64::max
  }
}

double score(const std::vector<double>& w, double b, const std::vector<double>& features) {
  double s = -0.0;
  for (std::size_t j = 0; j < w.size() && j < features.size(); ++j) s += w[j] * features[j];
  return s + b;
}

}  // namespace mi
