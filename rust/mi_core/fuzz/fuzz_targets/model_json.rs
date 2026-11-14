//! Any bytes into the model loader: it must return Ok or Err, never panic, and every model it
//! accepts must keep the contract's window limits (L >= 2, start and L at most 2^20 samples), a
//! rail_abs (when present) that is finite and > 0, and used channels in strictly ascending order.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mi_core::Model;

const MAX_WINDOW: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    // Model::from_path reads the file as UTF-8 first, so non-UTF-8 bytes never reach the parser.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(model) = Model::from_json(text) {
        assert!((2..=MAX_WINDOW).contains(&model.window_len()));
        assert!(model.window_start() <= MAX_WINDOW);
        assert!(!model.bands().is_empty());
        assert!(model.rail_abs().is_none_or(|r| r.is_finite() && r > 0.0));
        let used = model.used_channels();
        assert!(used.windows(2).all(|w| w[0] < w[1]) && used.iter().all(|&c| c < 8));
    }
});
