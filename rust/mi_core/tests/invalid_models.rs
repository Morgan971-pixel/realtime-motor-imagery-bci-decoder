//! Every invalid model in golden/invalid/ must be rejected, exactly as Python rejects it.

mod common;

use common::golden;
use mi_core::{CoreError, Model};
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    models: Vec<ModelCase>,
}

#[derive(Deserialize)]
struct ModelCase {
    file: String,
    reason: String,
}

#[test]
fn every_invalid_model_is_rejected() {
    let text = std::fs::read_to_string(golden("invalid/manifest.json")).unwrap();
    let manifest: Manifest = serde_json::from_str(&text).unwrap();
    // every invalid model on disk is listed (valid_* files are the valid spellings), and only those
    let mut on_disk: Vec<String> = std::fs::read_dir(golden("invalid"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.ends_with(".json") && f != "manifest.json" && !f.starts_with("valid_"))
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = manifest.models.iter().map(|c| c.file.clone()).collect();
    listed.sort();
    assert_eq!(listed, on_disk);
    for case in &manifest.models {
        let text = std::fs::read_to_string(golden(&format!("invalid/{}", case.file))).unwrap();
        match Model::from_json(&text) {
            Err(CoreError::Model(_)) => {}
            other => panic!(
                "{} ({}) was not rejected: {:?}",
                case.file,
                case.reason,
                other.map(|_| ())
            ),
        }
    }
}

#[test]
fn rejections_name_the_broken_rule() {
    for (file, needle) in [
        ("repeated_key.json", "duplicate field"),
        ("window_is_list.json", "must be a JSON object"),
        ("sos_a0_not_one_last_band.json", "a0 must equal 1"),
        ("lda_w_too_short.json", "lda.w"),
        ("band_high_at_nyquist.json", "fs/2"),
        ("window_len_huge.json", "spans more than"),
        ("fs_huge.json", "spans more than"),
    ] {
        let text = std::fs::read_to_string(golden(&format!("invalid/{file}"))).unwrap();
        let err = Model::from_json(&text).unwrap_err().to_string();
        assert!(err.contains(needle), "{file}: {err}");
    }
}
