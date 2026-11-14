//! Any bytes into the CSV reader: it must return Ok or Err, never panic, and a file it accepts
//! has at least one sample.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mi_core::csv::parse_csv;

fuzz_target!(|data: &[u8]| {
    // read_csv reads the file as UTF-8 first, so non-UTF-8 bytes never reach the parser.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(rows) = parse_csv(text) {
        assert!(!rows.is_empty());
    }
});
