//! Seconds to samples, exactly as the contract writes it: `floor(x + 0.5)`.

/// `seconds` and `fs` must already be validated as finite with `seconds >= 0`, `fs > 0`.
pub fn to_samples(seconds: f64, fs: f64) -> usize {
    debug_assert!(seconds.is_finite() && seconds >= 0.0 && fs.is_finite() && fs > 0.0);
    (seconds * fs + 0.5).floor() as usize
}

#[cfg(test)]
mod tests {
    use super::to_samples;

    #[test]
    fn default_window_at_250_hz() {
        assert_eq!(to_samples(0.5, 250.0), 125);
        assert_eq!(to_samples(2.0, 250.0), 500);
    }

    #[test]
    fn half_samples_round_up_like_the_python_reference() {
        // 0.5 s * 125 Hz = 62.5 samples; Python's built-in round() would say 62
        assert_eq!(to_samples(0.5, 125.0), 63);
    }

    #[test]
    fn half_sample_boundary_and_zero() {
        assert_eq!(to_samples(0.002, 250.0), 1); // exactly 0.5 samples rounds up
        assert_eq!(to_samples(0.0, 250.0), 0);
    }
}
