//! One band's causal filter: second-order sections in transposed direct form II (step 2).

use crate::model::N_CH;

#[derive(Debug, Clone)]
pub struct SosFilter {
    sections: Vec<[f64; 6]>,
    /// Per section, per channel: `[z0, z1]`. Starts at zero, never reset mid-stream.
    state: Vec<[[f64; 2]; N_CH]>,
}

impl SosFilter {
    pub fn new(sections: &[[f64; 6]]) -> Self {
        SosFilter {
            sections: sections.to_vec(),
            state: vec![[[0.0; 2]; N_CH]; sections.len()],
        }
    }

    /// Filter one multichannel sample. Same operations, same order, as the contract.
    pub fn process(&mut self, x: &[f64; N_CH]) -> [f64; N_CH] {
        let mut cur = *x;
        for (sec, z) in self.sections.iter().zip(self.state.iter_mut()) {
            let [b0, b1, b2, _a0, a1, a2] = *sec;
            for (c, zc) in z.iter_mut().enumerate() {
                let out = b0 * cur[c] + zc[0];
                zc[0] = b1 * cur[c] - a1 * out + zc[1];
                zc[1] = b2 * cur[c] - a2 * out;
                cur[c] = out;
            }
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_section_passes_input_through() {
        let mut f = SosFilter::new(&[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(f.process(&x), x);
    }

    #[test]
    fn filter_is_causal_and_keeps_state() {
        // y[n] = x[n] + x[n-1] (b0 = b1 = 1): state carries the previous input
        let mut f = SosFilter::new(&[[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]);
        assert_eq!(f.process(&[0.0; N_CH])[0], 0.0);
        assert_eq!(f.process(&[1.0; N_CH])[0], 1.0);
        assert_eq!(f.process(&[0.0; N_CH])[0], 1.0);
        assert_eq!(f.process(&[0.0; N_CH])[0], 0.0);
    }

    fn impulse(sections: &[[f64; 6]], n: usize) -> Vec<f64> {
        let mut f = SosFilter::new(sections);
        (0..n)
            .map(|k| f.process(&[if k == 0 { 1.0 } else { 0.0 }; N_CH])[0])
            .collect()
    }

    #[test]
    fn feedback_terms_have_the_contract_signs() {
        // y[n] = x[n] + 0.5 y[n-1]  (a1 = -0.5): powers of two keep the comparison exact
        assert_eq!(
            impulse(&[[1.0, 0.0, 0.0, 1.0, -0.5, 0.0]], 4),
            vec![1.0, 0.5, 0.25, 0.125]
        );
        // y[n] = x[n] + 0.25 y[n-2]  (a2 = -0.25)
        assert_eq!(
            impulse(&[[1.0, 0.0, 0.0, 1.0, 0.0, -0.25]], 5),
            vec![1.0, 0.0, 0.25, 0.0, 0.0625]
        );
    }

    #[test]
    fn b2_delays_by_two_samples_and_sections_cascade() {
        assert_eq!(
            impulse(&[[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], 4),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        // a one-sample delay followed by a two-sample delay is a three-sample delay
        let delays = [
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ];
        assert_eq!(impulse(&delays, 5), vec![0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn zero_input_gives_exactly_zero_output() {
        let mut f = SosFilter::new(&[[0.3, 0.1, -0.2, 1.0, -0.5, 0.25]; 4]);
        for _ in 0..100 {
            assert_eq!(f.process(&[0.0; N_CH]), [0.0; N_CH]);
        }
    }
}
