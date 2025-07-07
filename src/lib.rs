use pyo3::prelude::*;
use wide::*;
use numpy::{PyReadonlyArray1};

#[pyfunction]
fn first_true_1d_a(array: PyReadonlyArray1<bool>) -> isize {
    match array.as_slice() {
        Ok(slice) => slice.iter().position(|&v| v).map(|i| i as isize).unwrap_or(-1),
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
            if slice[i] { return i as isize; }
            if slice[i+1] { return (i+1) as isize; }
            if slice[i+2] { return (i+2) as isize; }
            if slice[i+3] { return (i+3) as isize; }
            if slice[i+4] { return (i+4) as isize; }
            if slice[i+5] { return (i+5) as isize; }
            if slice[i+6] { return (i+6) as isize; }
            if slice[i+7] { return (i+7) as isize; }
            i += 8;
        }

        // Handle remainder
        while i < len {
            if slice[i] { return i as isize; }
            i += 1;
        }
        -1
    } else {
        array.as_array().iter().position(|&v| v).map(|i| i as isize).unwrap_or(-1)
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
                if *slice.get_unchecked(i) { return i as isize; }
                if *slice.get_unchecked(i+1) { return (i+1) as isize; }
                if *slice.get_unchecked(i+2) { return (i+2) as isize; }
                if *slice.get_unchecked(i+3) { return (i+3) as isize; }
                if *slice.get_unchecked(i+4) { return (i+4) as isize; }
                if *slice.get_unchecked(i+5) { return (i+5) as isize; }
                if *slice.get_unchecked(i+6) { return (i+6) as isize; }
                if *slice.get_unchecked(i+7) { return (i+7) as isize; }
                i += 8;
            }

            // Handle remainder
            while i < len {
                if *slice.get_unchecked(i) { return i as isize; }
                i += 1;
            }
        }
        -1
    } else {
        array.as_array().iter().position(|&v| v).map(|i| i as isize).unwrap_or(-1)
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
        array.as_array().iter().position(|&v| v).map(|i| i as isize).unwrap_or(-1)
    }
}


#[pyfunction]
fn first_true_1d_f(py: Python, array: PyReadonlyArray1<bool>) -> isize {
    if let Ok(slice) = array.as_slice() {
        py.allow_threads(|| {
            let len = slice.len();
            let ptr = slice.as_ptr() as *const u8;
            let mut i = 0;

            let ones = u8x32::splat(1);
            unsafe {
                // Process 32 bytes at a time with SIMD
                while i + 32 <= len {
                    // Cast pointer to array reference
                    let bytes = &*(ptr.add(i) as *const [u8; 32]);

                    // Convert to SIMD vector
                    let chunk = u8x32::from(*bytes);
                    let equal_one = chunk.cmp_eq(ones);
                    if equal_one.any() {
                        break;
                    }

                    i += 32;
                }
                // // Handle final remainder
                while i < len.min(i + 32) {
                    if *ptr.add(i) != 0 {
                        return i as isize;
                    }
                    i += 1;
                }
                -1
            }
        })
    } else {
        let array_view = array.as_array();
        py.allow_threads(|| {
            array_view.iter().position(|&v| v).map(|i| i as isize).unwrap_or(-1)
        })
    }
}


#[pymodule]
fn arrayredox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(first_true_1d_a, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_b, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_c, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_d, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_e, m)?)?;
    m.add_function(wrap_pyfunction!(first_true_1d_f, m)?)?;
    Ok(())
}
