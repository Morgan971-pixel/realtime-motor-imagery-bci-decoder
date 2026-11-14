//! Club CSV reader: header `timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim`, one row per sample.
//!
//! Values are parsed as plain floats here (including "nan"); the decoder enforces the
//! stim and EEG rules, so a bad value is reported with its sample index.

use crate::error::{CoreError, Result};
use crate::model::N_CH;
use std::path::Path;

pub const HEADER: [&str; 10] = [
    "timestamps",
    "Fz",
    "C3",
    "Cz",
    "C4",
    "Pz",
    "PO7",
    "Oz",
    "PO8",
    "stim",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    pub sample: [f64; N_CH],
    pub stim: f64,
}

pub fn parse_csv(text: &str) -> Result<Vec<Row>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text); // Excel byte-order mark
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| CoreError::Csv("empty file".into()))?
        .split(',')
        .map(str::trim)
        .collect();
    if header != HEADER {
        return Err(CoreError::Csv(format!("unexpected columns {header:?}")));
    }
    let mut rows = Vec::new();
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != HEADER.len() {
            return Err(CoreError::Csv(format!(
                "line {}: expected {} columns, got {}",
                n + 2,
                HEADER.len(),
                fields.len()
            )));
        }
        let mut values = [0.0; 10];
        for ((v, f), column) in values.iter_mut().zip(&fields).zip(HEADER) {
            *v = f.trim().parse::<f64>().map_err(|_| {
                CoreError::Csv(format!(
                    "line {}, column {column}: not a number: {f:?}",
                    n + 2
                ))
            })?;
        }
        let mut sample = [0.0; N_CH];
        sample.copy_from_slice(&values[1..9]);
        rows.push(Row {
            sample,
            stim: values[9],
        });
    }
    if rows.is_empty() {
        return Err(CoreError::Csv("no samples".into()));
    }
    Ok(rows)
}

pub fn read_csv(path: impl AsRef<Path>) -> Result<Vec<Row>> {
    parse_csv(&std::fs::read_to_string(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim\n";

    #[test]
    fn reads_rows() {
        let rows = parse_csv(&format!("{HEAD}1.0,1,2,3,4,5,6,7,8,131\n")).unwrap();
        assert_eq!(
            rows,
            vec![Row {
                sample: [1., 2., 3., 4., 5., 6., 7., 8.],
                stim: 131.0
            }]
        );
    }

    #[test]
    fn tolerates_bom_and_spaces_in_header() {
        let text = format!("\u{feff}{}1.0,1,2,3,4,5,6,7,8,0\n", HEAD.replace(',', ", "));
        assert_eq!(parse_csv(&text).unwrap().len(), 1);
    }

    #[test]
    fn rejects_bad_files() {
        assert!(parse_csv("").is_err());
        assert!(parse_csv("time,a,b\n1,2,3\n").is_err());
        assert!(parse_csv(HEAD).is_err()); // header only
        assert!(parse_csv(&format!("{HEAD}1,2,3\n")).is_err()); // short row
        assert!(parse_csv(&format!("{HEAD}1,2,3,4,5,6,7,8,9,x\n")).is_err());
    }

    #[test]
    fn nan_parses_here_and_is_rejected_by_the_decoder() {
        let rows = parse_csv(&format!("{HEAD}1,2,3,4,5,6,7,8,9,nan\n")).unwrap();
        assert!(rows[0].stim.is_nan());
    }
}
