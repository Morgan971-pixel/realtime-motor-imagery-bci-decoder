// Log-variance features (step 5) and the LDA score (step 6).
#pragma once

#include <cstddef>
#include <vector>

#include "mi_core/model.hpp"
#include "mi_core/ring.hpp"

namespace mi {

/// A flat window gives ln(1e-12), never -inf.
constexpr double kVarFloor = 1e-12;

/// For each CSP row: project the window ending at sample i, then ln(max(population variance, floor)).
/// Two-pass variance: the mean first, then the sum of squared deviations. Appends to `out`.
void log_variance(const std::vector<Sample>& csp, const Ring& ring, std::size_t i, std::vector<double>& out);

/// score = sum(w[j] * f[j]) + b
double score(const std::vector<double>& w, double b, const std::vector<double>& features);

}  // namespace mi
