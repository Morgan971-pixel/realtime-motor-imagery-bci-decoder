//! Fixed-size window of the last L filtered samples (step 3): sample i goes in slot i % L.

use crate::model::N_CH;

#[derive(Debug, Clone)]
pub struct Ring {
    slots: Vec<[f64; N_CH]>,
}

impl Ring {
    /// Callers must pass `len >= 2`; the model loader guarantees it.
    pub fn new(len: usize) -> Self {
        assert!(len >= 2, "window must span at least 2 samples");
        Ring {
            slots: vec![[0.0; N_CH]; len],
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn put(&mut self, i: usize, value: [f64; N_CH]) {
        let n = self.slots.len();
        self.slots[i % n] = value;
    }

    /// The window ending at sample `i`, oldest first (the oldest is in slot (i + 1) % L).
    pub fn oldest_first(&self, i: usize) -> impl Iterator<Item = &[f64; N_CH]> + '_ {
        let n = self.slots.len();
        (0..n).map(move |k| &self.slots[(i + 1 + k) % n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_the_last_len_samples_oldest_first() {
        let mut r = Ring::new(3);
        for i in 0..7 {
            r.put(i, [i as f64; N_CH]);
        }
        let firsts: Vec<f64> = r.oldest_first(6).map(|s| s[0]).collect();
        assert_eq!(firsts, vec![4.0, 5.0, 6.0]);
    }
}
