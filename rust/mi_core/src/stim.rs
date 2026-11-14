//! Stim validation and marker detection (contract step 4).

pub const MAX_CODE: f64 = 2_147_483_647.0;
pub const IMAGERY_PHASE: u32 = 3;
pub const LEFT: u32 = 1;
pub const RIGHT: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marker {
    pub trial: u32,
    pub phase: u32,
    pub movement: u32,
}

impl Marker {
    pub fn is_imagery(&self) -> bool {
        self.phase == IMAGERY_PHASE && (self.movement == LEFT || self.movement == RIGHT)
    }
}

/// Check one raw stim value; returns the integer code (0 means no event).
pub fn validate_code(value: f64) -> Result<u32, String> {
    if !value.is_finite() {
        return Err("NaN or infinite value".into());
    }
    if value < 0.0 {
        return Err(format!("negative value {value}"));
    }
    if value > MAX_CODE {
        return Err(format!("value {value} above 2147483647"));
    }
    if value.fract() != 0.0 {
        return Err(format!("non-integer value {value}"));
    }
    let code = value as u32;
    if code != 0 && code < 100 {
        return Err(format!("code {code} below 100 (trial >= 1 required)"));
    }
    Ok(code)
}

/// Split a validated nonzero code into trial, phase and movement (integer division).
pub fn parse(code: u32) -> Marker {
    debug_assert!(code >= 100, "parse needs a validated nonzero code");
    Marker {
        trial: code / 100,
        phase: (code / 10) % 10,
        movement: code % 10,
    }
}

/// A nonzero code is a new marker unless the previous sample held the same code.
#[derive(Debug, Default, Clone)]
pub struct MarkerTracker {
    prev: u32,
}

impl MarkerTracker {
    /// Call only with codes that passed `validate_code`, after the whole sample was validated.
    pub fn observe(&mut self, code: u32) -> Option<Marker> {
        let is_new = code != 0 && code != self.prev;
        self.prev = code;
        is_new.then(|| parse(code))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_splits_trial_phase_movement() {
        assert_eq!(
            parse(2131),
            Marker {
                trial: 21,
                phase: 3,
                movement: 1
            }
        );
        assert_eq!(
            parse(652),
            Marker {
                trial: 6,
                phase: 5,
                movement: 2
            }
        );
    }

    #[test]
    fn invalid_values_are_rejected() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            -1.0,
            131.5,
            31.0,
            99.0,
            2_147_483_648.0,
        ] {
            assert!(validate_code(bad).is_err(), "{bad} should be rejected");
        }
        assert_eq!(validate_code(0.0), Ok(0));
        assert_eq!(validate_code(-0.0), Ok(0));
        assert_eq!(validate_code(100.0), Ok(100));
        assert_eq!(validate_code(131.0), Ok(131));
        assert_eq!(validate_code(MAX_CODE), Ok(2_147_483_647));
        assert!(validate_code(2_147_483_647.5)
            .unwrap_err()
            .contains("above"));
    }

    fn markers(codes: &[u32]) -> Vec<usize> {
        let mut t = MarkerTracker::default();
        codes
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| t.observe(c).map(|_| i))
            .collect()
    }

    #[test]
    fn runs_follow_the_contract() {
        assert_eq!(markers(&[131, 131, 131]), vec![0]); // A,A,A: one marker
        assert_eq!(markers(&[131, 0, 131]), vec![0, 2]); // A,0,A: two
        assert_eq!(markers(&[131, 151, 131]), vec![0, 1, 2]); // A,B,A: three
        assert_eq!(markers(&[131, 151, 151]), vec![0, 1]); // A,B,B: two
        assert_eq!(markers(&[0, 0, 131]), vec![2]);
    }

    #[test]
    fn tracker_returns_the_parsed_marker() {
        let mut t = MarkerTracker::default();
        assert_eq!(t.observe(232), Some(parse(232)));
        assert_eq!(t.observe(232), None);
        assert_eq!(
            parse(2_147_483_647),
            Marker {
                trial: 21_474_836,
                phase: 4,
                movement: 7
            }
        );
    }

    #[test]
    fn only_phase_3_left_or_right_is_imagery() {
        assert!(parse(131).is_imagery());
        assert!(parse(232).is_imagery());
        assert!(!parse(121).is_imagery());
        assert!(!parse(141).is_imagery());
        assert!(!parse(133).is_imagery());
    }
}
