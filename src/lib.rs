use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

#[pyfunction]
fn first_true_1d(array: PyReadonlyArray1<bool>) -> isize {
    match array.as_slice() {
        Ok(slice) => slice.iter().position(|&v| v).map(|i| i as isize).unwrap_or(-1),
        Err(_) => -1, // Should not happen for 1D arrays, but fallback to -1
    }
}


#[pymodule]
fn arrayredox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(first_true_1d, m)?)?;
    Ok(())
}
