use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;
// use pyo3::types::{PyBool, PyAny};
use wide::*;
// use std::simd::Simd;
// use std::simd::cmp::SimdPartialEq;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use numpy::ToPyArray;
use numpy::{PyArray2, PyReadonlyArray2};
use numpy::PyArray1;

#[pyfunction]
fn first_true_1d_a(array: PyReadonlyArray1<bool>) -> isize {
    match array.as_slice() {
        Ok(slice) => slice
            .iter()
            .position(|&v| v)
            .map(|i| i as isize)
            .unwrap_or(-1),
        Err(_) => -1, // Should not happen for 1D arrays, but fallback to -1
    }
}

// Release the GIL, still doing slice iteration
#[pyfunction]
fn first_true_1d_b(py: Python, array: PyReadonlyArray1<bool>) -> isize {
    if let Ok(slice) = array.as_slice() {
        py.allow_threads(|| {
            for (i, &value) in slice.iter().enumerate() {
                if value {
                    return i as isize;
                }
            }
            -1
        })
    } else {
        let array_view = array.as_array();
        py.allow_threads(|| {
            for (idx, &val) in array_view.iter().enumerate() {
                if val {
                    return idx as isize;
                }
            }
            -1
        })
    }
}

#[pyfunction]
fn first_true_1d_c(array: PyReadonlyArray1<bool>) -> isize {
    if let Ok(slice) = array.as_slice() {
        let len = slice.len();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= len {
            if slice[i] {
                return i as isize;
            }
            if slice[i + 1] {
                return (i + 1) as isize;
            }
            if slice[i + 2] {
                return (i + 2) as isize;
            }
            if slice[i + 3] {
                return (i + 3) as isize;
            }
            if slice[i + 4] {
                return (i + 4) as isize;
            }
            if slice[i + 5] {
                return (i + 5) as isize;
            }
            if slice[i + 6] {
                return (i + 6) as isize;
            }
            if slice[i + 7] {
                return (i + 7) as isize;
            }
            i += 8;
        }

        // Handle remainder
        while i < len {
            if slice[i] {
                return i as isize;
            }
            i += 1;
        }
        -1
    } else {
        array
            .as_array()
            .iter()
            .position(|&v| v)
            .map(|i| i as isize)
            .unwrap_or(-1)
    }
}

#[pyfunction]
fn first_true_1d_d(array: PyReadonlyArray1<bool>) -> isize {
    if let Ok(slice) = array.as_slice() {
        let len = slice.len();
        let mut i = 0;

        unsafe {
            // Process 8 elements at a time
            while i + 8 <= len {
                if *slice.get_unchecked(i) {
                    return i as isize;
                }
                if *slice.get_unchecked(i + 1) {
                    return (i + 1) as isize;
                }
                if *slice.get_unchecked(i + 2) {
                    return (i + 2) as isize;
                }
                if *slice.get_unchecked(i + 3) {
                    return (i + 3) as isize;
                }
                if *slice.get_unchecked(i + 4) {
                    return (i + 4) as isize;
                }
                if *slice.get_unchecked(i + 5) {
                    return (i + 5) as isize;
                }
                if *slice.get_unchecked(i + 6) {
                    return (i + 6) as isize;
                }
                if *slice.get_unchecked(i + 7) {
                    return (i + 7) as isize;
                }
                i += 8;
            }

            // Handle remainder
            while i < len {
                if *slice.get_unchecked(i) {
                    return i as isize;
                }
                i += 1;
            }
        }
        -1
    } else {
        array
            .as_array()
            .iter()
            .position(|&v| v)
            .map(|i| i as isize)
            .unwrap_or(-1)
    }
}

#[pyfunction]
fn first_true_1d_e(array: PyReadonlyArray1<bool>) -> isize {
    if let Ok(slice) = array.as_slice() {
        let len = slice.len();
        let ptr = slice.as_ptr() as *const u8;

        unsafe {
            // Process 8 bytes at a time as u64
            let mut i = 0;
            while i + 8 <= len {
                // Check 8 bytes at once
                let chunk = *(ptr.add(i) as *const u64);
                if chunk != 0 {
                    // Found a true value in this chunk, check each byte
                    for j in 0..8 {
                        if i + j < len && *ptr.add(i + j) != 0 {
                            return (i + j) as isize;
                        }
                    }
                }
                i += 8;
            }

            // Handle remainder
            while i < len {
                if *ptr.add(i) != 0 {
                    return i as isize;
                }
                i += 1;
            }
        }
        -1
    } else {
        array
            .as_array()
            .iter()
            .position(|&v| v)
            .map(|i| i as isize)
            .unwrap_or(-1)
    }
}

#[pyfunction]
#[pyo3(signature = (array, forward=true))]
fn first_true_1d(py: Python, array: PyReadonlyArray1<bool>, forward: bool) -> isize {
    if let Ok(slice) = array.as_slice() {
        const LANES: usize = 32;

        py.allow_threads(|| {
            let len = slice.len();
            let ptr = slice.as_ptr() as *const u8;
            let ones = u8x32::splat(1);

            if forward {
                let mut i = 0;
                unsafe {
                    // Process 32 bytes at a time with SIMD
                    while i + LANES <= len {
                        let bytes = &*(ptr.add(i) as *const [u8; LANES]);
                        let chunk = u8x32::from(*bytes);
                        let equal_one = chunk.cmp_eq(ones);
                        if equal_one.any() {
                            break;
                        }
                        i += LANES;
                    }
                    // Handle final remainder
                    while i < len.min(i + LANES) {
                        if *ptr.add(i) != 0 {
                            return i as isize;
                        }
                        i += 1;
                    }
                }
            } else {
                // Backward search
                let mut i = len;
                unsafe {
                    // Process LANES bytes at a time with SIMD (backwards)
                    while i >= LANES {
                        i -= LANES;
                        let bytes = &*(ptr.add(i) as *const [u8; LANES]);
                        let chunk = u8x32::from(*bytes);
                        let equal_one = chunk.cmp_eq(ones);
                        if equal_one.any() {
                            // Found a true in this chunk, search backwards within it
                            for j in (i..i + LANES).rev() {
                                if *ptr.add(j) != 0 {
                                    return j as isize;
                                }
                            }
                        }
                    }
                    // Handle remaining bytes at the beginning
                    if i > 0 {
                        for j in (0..i).rev() {
                            if *ptr.add(j) != 0 {
                                return j as isize;
                            }
                        }
                    }
                }
            }
            -1
        })
    } else {
        let array_view = array.as_array();
        py.allow_threads(|| {
            if forward {
                array_view
                    .iter()
                    .position(|&v| v)
                    .map(|i| i as isize)
                    .unwrap_or(-1)
            } else {
                array_view
                    .iter()
                    .rposition(|&v| v)
                    .map(|i| i as isize)
                    .unwrap_or(-1)
            }
        })
    }
}

// #[pyfunction]
// fn first_true_1d_g(py: Python, array: PyReadonlyArray1<bool>) -> isize {
//     if let Ok(slice) = array.as_slice() {
//         py.allow_threads(|| {
//             let len = slice.len();
//             let ptr = slice.as_ptr() as *const u8;
//             let mut i = 0;

//             type Lane = u8;
//             const LANES: usize = 64;
//             let ones = Simd::<Lane, LANES>::splat(1);

//             unsafe {
//                 while i + LANES <= len {
//                     let chunk_ptr = ptr.add(i) as *const [u8; LANES];
//                     let chunk = Simd::from(*chunk_ptr);
//                     let mask = chunk.simd_eq(ones).to_bitmask();

//                     if mask != 0 {
//                         let offset = mask.trailing_zeros() as usize;
//                         return (i + offset) as isize;
//                     }

//                     i += LANES;
//                 }

//                 // Remainder (non-SIMD tail)
//                 while i < len {
//                     if *ptr.add(i) != 0 {
//                         return i as isize;
//                     }
//                     i += 1;
//                 }
//             }

//             -1
//         })
//     } else {
//         // Fallback for non-contiguous arrays
//         let view = array.as_array();
//         py.allow_threads(|| {
//             view.iter()
//                 .position(|&v| v)
//                 .map(|i| i as isize)
//                 .unwrap_or(-1)
//         })
//     }
// }

//------------------------------------------------------------------------------


// NOTE: we copy the entire array into contiguous memory when necessary.
// axis = 0 returns the pos per col
// axis = 1 returns the pos per row (as contiguous bytes)
// if c contiguous:
//      axis == 0: transpose, copy to C
//      axis == 1: keep
// if f contiguous:
//      axis == 0: transpose, keep
//      axis == 1: copy to C
// else
//     axis == 0: transpose, copy to C
//     axis == 1: copy to C

fn prepare_array_for_axis<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, bool>,
    axis: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    if axis != 0 && axis != 1 {
        return Err(PyValueError::new_err("axis must be 0 or 1"));
    }

    let is_c = array.is_c_contiguous();
    let is_f = array.is_fortran_contiguous();

    match (is_c, is_f, axis) {
        (true, _, 0) => Ok(array.as_array().reversed_axes().to_pyarray(py)),
        (true, _, 1) => Ok(array.as_array().to_pyarray(py)),
        (_, true, 0) => Ok(array.as_array().reversed_axes().to_pyarray(py)),
        (_, true, 1) => Ok(array.as_array().to_pyarray(py)),
        (_, _, 0) => Ok(array.as_array().reversed_axes().to_pyarray(py)),
        (_, _, 1) => Ok(array.as_array().to_pyarray(py)),
        _ => unreachable!(),
    }
}

#[pyfunction]
pub fn first_true_2d<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, bool>,
    axis: usize,
) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let prepped = prepare_array_for_axis(py, array, axis)?;
    let view = unsafe { prepped.as_array() };

    // NOTE: these are rows in the view, not always the same as rows
    let rows = view.nrows();
    let mut result = Vec::with_capacity(rows);

    py.allow_threads(|| {
        const LANES: usize = 32;
        let ones = u8x32::splat(1);

        for row in 0..rows {
            let mut found = -1;
            let row_slice = &view.row(row);
            let ptr = row_slice.as_ptr() as *const u8;
            let len = row_slice.len();
            let mut i = 0;

            unsafe {
                while i + LANES <= len {
                    let chunk = &*(ptr.add(i) as *const [u8; LANES]);
                    let vec = u8x32::from(*chunk);
                    if vec.cmp_eq(ones).any() {
                        break;
                    }
                    i += LANES;
                }
                while i < len {
                    if *ptr.add(i) != 0 {
                        found = i as isize;
                        break;
                    }
                    i += 1;
                }
            }
            result.push(found);
        }
    });

    Ok(PyArray1::from_vec(py, result).to_owned())
}


//------------------------------------------------------------------------------

#[pymodule]
fn arrayredox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(first_true_1d_a, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_b, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_c, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_d, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_e, m)?)?;
    // m.add_function(wrap_pyfunction!(first_true_1d_g, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_2d, m)?)?;
    Ok(())
}
