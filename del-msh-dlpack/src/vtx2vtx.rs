use crate::slice_shape_from_tensor;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(vtx2vtx_laplacian_smoothing, m)?)?;
    Ok(())
}

/// Pythonから渡された PyCapsule を Rust 側で読み取る
#[pyo3::pyfunction]
fn vtx2vtx_laplacian_smoothing(
    _py: Python,
    vtx2idx: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    lambda: f32,
    vtx2lhs: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    num_iter: usize,
    vtx2lhstmp: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2idx = crate::get_managed_tensor_from_pyany(vtx2idx)?;
    let idx2vtx = crate::get_managed_tensor_from_pyany(idx2vtx)?;
    let vtx2rhs = crate::get_managed_tensor_from_pyany(vtx2rhs)?;
    let vtx2lhs = crate::get_managed_tensor_from_pyany(vtx2lhs)?;
    let vtx2lhstmp = crate::get_managed_tensor_from_pyany(vtx2lhstmp)?;
    //
    let vtx2idx_shape = unsafe { std::slice::from_raw_parts(vtx2idx.shape, vtx2idx.ndim as usize) };
    let vtx2lhs_shape = unsafe { std::slice::from_raw_parts(vtx2lhs.shape, vtx2lhs.ndim as usize) };
    let num_vtx = vtx2idx_shape[0] - 1;
    let num_dim = vtx2lhs_shape[1];
    assert_eq!(vtx2idx.dtype.code, 1);
    assert_eq!(vtx2idx.dtype.bits, 64);
    assert_eq!(idx2vtx.dtype.code, 1);
    assert_eq!(idx2vtx.dtype.bits, 64);
    assert_eq!(vtx2lhs.dtype.code, 2);
    assert_eq!(vtx2lhs.dtype.bits, 32);
    assert_eq!(vtx2rhs.dtype.code, 2);
    assert_eq!(vtx2rhs.dtype.bits, 32);
    assert_eq!(vtx2lhstmp.dtype.code, 2);
    assert_eq!(vtx2lhstmp.dtype.bits, 32);

    match vtx2idx.ctx.device_type {
        dlpack::device_type_codes::CPU => {
            use crate::slice_shape_from_tensor_mut;
            let (vtx2idx, vtx2idx_sh) =
                unsafe { slice_shape_from_tensor::<usize>(vtx2idx).unwrap() };
            assert_eq!(vtx2idx_sh, vec!(num_vtx + 1));
            let (idx2vtx, idx2vtx_sh) =
                unsafe { slice_shape_from_tensor::<usize>(idx2vtx).unwrap() };
            assert_eq!(idx2vtx_sh.len(), 1);
            let (vtx2rhs, vtx2rhs_sh) = unsafe { slice_shape_from_tensor::<f32>(vtx2rhs).unwrap() };
            assert_eq!(vtx2rhs_sh, vec!(num_vtx, num_dim));
            let (vtx2lhs, vtx2lhs_sh) =
                unsafe { slice_shape_from_tensor_mut::<f32>(vtx2lhs).unwrap() };
            assert_eq!(vtx2lhs_sh, vec!(num_vtx, num_dim));
            let (vtx2lhstmp, vtx2lhstmp_sh) =
                unsafe { slice_shape_from_tensor_mut::<f32>(vtx2lhstmp).unwrap() };
            assert_eq!(vtx2lhstmp_sh, vec!(num_vtx, num_dim));
            del_msh_cpu::vtx2vtx::laplacian_smoothing::<3>(
                vtx2idx, idx2vtx, lambda, vtx2lhs, vtx2rhs, num_iter, vtx2lhstmp,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            println!("GPU_{}", vtx2idx.ctx.device_id);
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::LAPLACIAN_SMOOTHING_JACOBI,
                "laplacian_smoothing_jacobi",
            );
            let stream = del_cudarc_sys::create_stream_in_current_context();
            for _itr in 0..num_iter {
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_i32(num_vtx as i32);
                    builder.arg_data(&vtx2idx.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2lhstmp.data);
                    builder.arg_data(&vtx2xyz.data);
                    builder.arg_data(&rhs.data);
                    unsafe {
                        builder.launch_kernel(function, num_vtx as u32);
                    }
                }
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_i32(num_vtx as i32);
                    builder.arg_data(&vtx2idx.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2xyz.data);
                    builder.arg_data(&vtx2lhstmp.data);
                    builder.arg_data(&rhs.data);
                    unsafe {
                        builder.launch_kernel(function, num_vtx as u32);
                    }
                }
            }
            unsafe {
                del_cudarc_sys::cuStreamSynchronize(stream);
                del_cudarc_sys::cuStreamDestroy_v2(stream);
            }
        }
        _ => println!("Unknown device type {}", vtx2idx.ctx.device_type),
    }
    Ok(())
}
