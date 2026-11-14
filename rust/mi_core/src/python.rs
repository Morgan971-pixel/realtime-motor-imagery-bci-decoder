//! Python bindings (feature "python"): `import mi_core`.
//!
//! Every CoreError becomes a Python ValueError with the same message.

// PyO3 0.22's #[pyfunction]/#[pymethods] expansions trip this lint on PyResult signatures;
// the conversion is in generated code, not ours.
#![allow(clippy::useless_conversion)]

use crate::csv::read_csv;
use crate::{decode, CoreError, Decision, Decoder, Model};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py(e: CoreError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn decision_dict(py: Python<'_>, d: &Decision) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("sample", d.sample)?;
    out.set_item("trial", d.trial)?;
    out.set_item("true_label", d.true_label)?;
    out.set_item("features", d.features.clone())?;
    out.set_item("score", d.score)?;
    out.set_item("label", d.label)?;
    out.set_item("pinned", d.pinned.clone())?;
    Ok(out.into())
}

/// Streaming decoder: `Decoder(model_json_text).push(sample, stim) -> dict | None`.
/// `sample` is any sequence of 8 numbers (list, tuple, NumPy row): a wrong length raises
/// ValueError, non-numbers raise TypeError, NaN/inf reach the core and raise ValueError.
#[pyclass(name = "Decoder")]
struct PyDecoder {
    inner: Decoder,
}

#[pymethods]
impl PyDecoder {
    #[new]
    fn new(model_json: &str) -> PyResult<Self> {
        let model = Model::from_json(model_json).map_err(to_py)?;
        Ok(PyDecoder {
            inner: Decoder::new(model),
        })
    }

    fn push(&mut self, py: Python<'_>, sample: [f64; 8], stim: f64) -> PyResult<Option<PyObject>> {
        match self.inner.push(&sample, stim).map_err(to_py)? {
            Some(d) => Ok(Some(decision_dict(py, &d)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn samples_seen(&self) -> usize {
        self.inner.samples_seen()
    }
}

/// Validate a model.json text; raises ValueError if it breaks the contract.
#[pyfunction]
fn check_model(model_json: &str) -> PyResult<()> {
    Model::from_json(model_json).map(|_| ()).map_err(to_py)
}

/// Decode a whole club CSV recording with a model.json file; returns a list of dicts.
/// Releases the GIL while reading and decoding, so other Python threads keep running.
#[pyfunction]
fn decode_file(py: Python<'_>, model_path: &str, csv_path: &str) -> PyResult<Vec<PyObject>> {
    let at = |path: &str, e: CoreError| PyValueError::new_err(format!("{path}: {e}"));
    let decisions = py.allow_threads(|| -> PyResult<Vec<Decision>> {
        let model = Model::from_path(model_path).map_err(|e| at(model_path, e))?;
        let rows = read_csv(csv_path).map_err(|e| at(csv_path, e))?;
        decode(&model, &rows).map_err(|e| at(csv_path, e))
    })?;
    decisions.iter().map(|d| decision_dict(py, d)).collect()
}

#[pymodule]
fn mi_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDecoder>()?;
    m.add_function(wrap_pyfunction!(check_model, m)?)?;
    m.add_function(wrap_pyfunction!(decode_file, m)?)?;
    Ok(())
}
