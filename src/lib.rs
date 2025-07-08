use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::Bound;
// use pyo3::types::{PyBool, PyAny};
use wide::*;
// use std::simd::Simd;
// use std::simd::cmp::SimdPartialEq;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use numpy::ToPyArray;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;


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
fn first_true_1d(py: Python,
    array: PyReadonlyArray1<bool>,
    forward: bool,
) -> isize {
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
        (true, _, 0) => {
            let transposed = array.as_array().reversed_axes().to_owned();
            Ok(transposed.into_pyarray(py))
        }
        (true, _, 1) => Ok(array.as_array().to_owned().into_pyarray(py)),  // copy to get full ownership
        (_, true, 0) => {
            let transposed = array.as_array().reversed_axes();
            Ok(transposed.to_owned().into_pyarray(py))
        }
        (_, true, 1) => {
            let owned = array.as_array().to_owned();
            Ok(owned.into_pyarray(py))
        }
        (false, false, 0) => {
            let transposed = array.as_array().reversed_axes().to_owned();
            Ok(transposed.into_pyarray(py))
        }
        (false, false, 1) => {
            let owned = array.as_array().to_owned();
            Ok(owned.into_pyarray(py))
        }
        _ => unreachable!(),
    }
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
    Ok(())
}
