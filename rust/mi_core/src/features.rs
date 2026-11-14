//! Log-variance features (step 5) and the LDA score (step 6).

use crate::model::N_CH;
use crate::ring::Ring;

/// A flat window gives ln(1e-12), never -inf.
pub const VAR_FLOOR: f64 = 1e-12;

fn dot(w: &[f64; N_CH], x: &[f64; N_CH]) -> f64 {
    let mut acc = 0.0;
    for c in 0..N_CH {
        acc += w[c] * x[c];
    }
    acc
}

/// For each CSP row: project the window, then ln(max(population variance, floor)).
/// Two-pass variance: the mean first, then the sum of squared deviations.
pub fn log_variance(csp: &[[f64; N_CH]], ring: &Ring, i: usize, out: &mut Vec<f64>) {
    let len = ring.len() as f64;
    let mut z = Vec::with_capacity(ring.len());
    for w in csp {
        z.clear();
        z.extend(ring.oldest_first(i).map(|x| dot(w, x)));
        let mean = z.iter().sum::<f64>() / len;
        let ss: f64 = z.iter().map(|v| (v - mean) * (v - mean)).sum();
        out.push((ss / len).max(VAR_FLOOR).ln());
    }
}

/// score = sum(w[j] * f[j]) + b
pub fn score(w: &[f64], b: f64, features: &[f64]) -> f64 {
    w.iter().zip(features).map(|(wj, fj)| wj * fj).sum::<f64>() + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_explicit_formula() {
        let mut ring = Ring::new(4);
        let vals = [1.0, 3.0, 2.0, 6.0];
        for (i, v) in vals.iter().enumerate() {
            let mut s = [0.0; N_CH];
            s[0] = *v;
            ring.put(i, s);
        }
        let mut row = [0.0; N_CH];
        row[0] = 1.0;
        let mut out = Vec::new();
        log_variance(&[row], &ring, 3, &mut out);
        // mean 3, squared deviations 4 + 0 + 1 + 9 = 14, population variance 3.5
        assert!((out[0] - 3.5f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn flat_window_hits_the_floor() {
        let ring = Ring::new(5);
        let mut out = Vec::new();
        log_variance(&[[1.0; N_CH]], &ring, 4, &mut out);
        assert_eq!(out, vec![VAR_FLOOR.ln()]);
    }

    #[test]
    fn score_is_dot_plus_bias() {
        assert_eq!(score(&[1.0, -2.0], 0.5, &[3.0, 1.0]), 1.5);
    }
}
