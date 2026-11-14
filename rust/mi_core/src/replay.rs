//! Replay a recording as if a headset were streaming it, timing every decision.
//!
//! The same loop as `mi_ref.replay` and the C++ core (docs/replay.md): fast mode hands each
//! chunk over immediately; paced mode first sleeps until the chunk's last sample is due at
//! the model's sampling rate. A decision's latency runs from its chunk's handover (when the
//! loop is awake and starts pushing) to the moment `push` returns it.

use crate::csv::Row;
use crate::decoder::{Decision, Decoder};
use crate::error::Result;
use crate::model::Model;
use std::time::{Duration, Instant};

/// Paced sleep targets stop growing here (2^63 - 1 ns, about 292 years), as in the C++ core.
const MAX_DUE_NS: u64 = i64::MAX as u64;

/// Monotonic time in nanoseconds, injectable so tests can replay with a fake clock.
pub trait Clock {
    fn now_ns(&mut self) -> u64;
    fn sleep_until_ns(&mut self, target: u64);
}

/// Monotonic nanoseconds since the clock was created.
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        SystemClock {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_ns(&mut self) -> u64 {
        self.origin.elapsed().as_nanos() as u64 // u64 nanoseconds last 584 years
    }
    fn sleep_until_ns(&mut self, target: u64) {
        let now = self.now_ns();
        if target > now {
            std::thread::sleep(Duration::from_nanos(target - now));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayOptions {
    /// Samples handed over together (a headset delivers small blocks, not single samples).
    pub chunk: usize,
    /// Deliver chunks at the model's sampling rate instead of as fast as possible.
    pub paced: bool,
    /// Play the recording this many times back to back.
    pub repeat: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Replay {
    pub decisions: Vec<Decision>,
    /// One per decision.
    pub latencies_ns: Vec<u64>,
    pub samples: usize,
    pub elapsed_ns: u64,
}

/// Panics if `chunk` or `repeat` is 0. Invalid rows return the decoder's error.
pub fn replay(
    model: &Model,
    rows: &[Row],
    opts: ReplayOptions,
    clock: &mut impl Clock,
) -> Result<Replay> {
    assert!(
        opts.chunk >= 1 && opts.repeat >= 1,
        "chunk and repeat must be at least 1, got chunk={}, repeat={}",
        opts.chunk,
        opts.repeat
    );
    let mut dec = Decoder::new(model.clone());
    let n = rows.len();
    let total = n
        .checked_mul(opts.repeat)
        .expect("recording too long to repeat that many times");
    let mut decisions = Vec::new();
    let mut latencies_ns = Vec::new();
    let start = clock.now_ns();
    for first in (0..total).step_by(opts.chunk) {
        let last = (first + opts.chunk).min(total);
        if opts.paced {
            // `as u64` saturates; the clamp only matters past ~292 years of samples
            let due_ns = (last as f64 * 1e9 / model.fs()) as u64;
            clock.sleep_until_ns(start.saturating_add(due_ns.min(MAX_DUE_NS)));
        }
        let handover = clock.now_ns();
        for j in first..last {
            let row = &rows[j % n];
            if let Some(d) = dec.push(&row.sample, row.stim)? {
                latencies_ns.push(clock.now_ns() - handover);
                decisions.push(d);
            }
        }
    }
    let elapsed_ns = clock.now_ns() - start;
    Ok(Replay {
        decisions,
        latencies_ns,
        samples: total,
        elapsed_ns,
    })
}
