
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray4, PyUntypedArrayMethods};
use rayon::prelude::*;

#[pyfunction]
fn im2col<'py>(
    py: Python<'py>,
    x: PyReadonlyArray4<'py, f32>,
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
) -> Bound<'py, PyArray2<f32>> {

    let n = x.shape()[0];
    let c = x.shape()[1];
    let h = x.shape()[2];
    let w = x.shape()[3];
    let x = x.as_array();
    let x_slice = x.as_slice().expect("вход должен быть непрерывным (C-order)");

    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    let rows = c * kh * kw;
    let cols = n * out_h * out_w;

    let mut out = vec![0.0f32; rows * cols];

    let out_ptr = out.as_mut_ptr() as usize;

    (0..n).into_par_iter().for_each(|ni| {
        let out_base = out_ptr as *mut f32;

        for row_idx in 0..rows {
            let ci = row_idx / (kh * kw);
            let rem = row_idx % (kh * kw);
            let ki = rem / kw;
            let kj = rem % kw;

            for oh in 0..out_h {
                let ih = oh * stride + ki;
                if ih < padding || (ih - padding) >= h {
                    continue;
                }
                let ih_real = ih - padding;

                let in_row_base = ((ni * c + ci) * h + ih_real) * w;

                for ow in 0..out_w {
                    let iw = ow * stride + kj;
                    if iw < padding || (iw - padding) >= w {
                        continue;
                    }
                    let iw_real = iw - padding;

                    let val = x_slice[in_row_base + iw_real];

                    let col = (oh * out_w + ow) * n + ni;
                    let out_idx = row_idx * cols + col;

                    unsafe {
                        *out_base.add(out_idx) = val;
                    }
                }
            }
        }
    });

    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), out)
        .expect("не удалось собрать матрицу");

    PyArray2::from_owned_array(py, arr)
}

#[pymodule]
fn tinytensor_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(im2col, m)?)?;
    Ok(())
}
