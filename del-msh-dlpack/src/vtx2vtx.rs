use pyo3::{pyfunction, types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(vtx2vtx_laplacian_smoothing, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vtx2vtx_multiply_graph_laplacian, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vtx2vtx_from_uniform_mesh, m)?)?;
    Ok(())
}

/// Solve the linear system from screened Poisson equation using Jacobi method:
///
/// \[I + lambda * L\] {vtx2lhs} = {vtx2rhs}
/// where L = \[ .., -1, .., valence, ..,-1, .. \]
#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
fn vtx2vtx_laplacian_smoothing(
    _py: Python,
    vtx2idx: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    lambda: f32,
    vtx2lhs: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    num_iter: usize,
    vtx2lhstmp: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
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
    assert_eq!(vtx2idx.dtype.code, dlpack::data_type_codes::UINT);
    assert_eq!(vtx2idx.dtype.bits, 32, "index type must be u32");
    assert_eq!(idx2vtx.dtype.code, dlpack::data_type_codes::UINT);
    assert_eq!(idx2vtx.dtype.bits, 32, "index type must be u32");
    assert_eq!(vtx2lhs.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(vtx2lhs.dtype.bits, 32);
    assert_eq!(vtx2rhs.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(vtx2rhs.dtype.bits, 32);
    assert_eq!(vtx2lhstmp.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(vtx2lhstmp.dtype.bits, 32);
    //
    let device_type = vtx2idx.ctx.device_type;
    assert_eq!(idx2vtx.ctx.device_type, device_type);
    assert_eq!(vtx2rhs.ctx.device_type, device_type);
    assert_eq!(vtx2lhs.ctx.device_type, device_type);
    assert_eq!(vtx2lhstmp.ctx.device_type, device_type);

    match device_type {
        dlpack::device_type_codes::CPU => {
            let (vtx2idx, vtx2idx_sh) =
                unsafe { crate::slice_shape_from_tensor::<u32>(vtx2idx).unwrap() };
            assert_eq!(vtx2idx_sh, vec!(num_vtx + 1));
            let (idx2vtx, idx2vtx_sh) =
                unsafe { crate::slice_shape_from_tensor::<u32>(idx2vtx).unwrap() };
            assert_eq!(idx2vtx_sh.len(), 1);
            let (vtx2rhs, vtx2rhs_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(vtx2rhs).unwrap() };
            assert_eq!(vtx2rhs_sh, vec!(num_vtx, num_dim));
            let (vtx2lhs, vtx2lhs_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<f32>(vtx2lhs).unwrap() };
            assert_eq!(vtx2lhs_sh, vec!(num_vtx, num_dim));
            let (vtx2lhstmp, vtx2lhstmp_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<f32>(vtx2lhstmp).unwrap() };
            assert_eq!(vtx2lhstmp_sh, vec!(num_vtx, num_dim));
            del_msh_cpu::vtx2vtx::laplacian_smoothing::<3, u32>(
                vtx2idx, idx2vtx, lambda, vtx2lhs, vtx2rhs, num_iter, vtx2lhstmp,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            // println!("GPU_{}", vtx2idx.ctx.device_id);
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::VTX2VTX,
                "laplacian_smoothing",
            );
            unsafe {
                del_cudarc_sys::cuInit(0);
            }
            assert_eq!(std::mem::size_of::<usize>(), 8);
            let stream = stream_ptr as usize as *mut std::ffi::c_void as del_cudarc_sys::CUstream;
            assert_eq!(vtx2idx.byte_offset, 0);
            assert_eq!(idx2vtx.byte_offset, 0);
            assert_eq!(vtx2lhstmp.byte_offset, 0);
            assert_eq!(vtx2lhs.byte_offset, 0);
            assert_eq!(vtx2rhs.byte_offset, 0);
            for _itr in 0..num_iter {
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_i32(num_vtx as i32);
                    builder.arg_data(&vtx2idx.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2lhstmp.data);
                    builder.arg_data(&vtx2lhs.data);
                    builder.arg_data(&vtx2rhs.data);
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
                    builder.arg_data(&vtx2lhs.data);
                    builder.arg_data(&vtx2lhstmp.data);
                    builder.arg_data(&vtx2rhs.data);
                    unsafe {
                        builder.launch_kernel(function, num_vtx as u32);
                    }
                }
            }
            unsafe {
                let res_sync = del_cudarc_sys::cuStreamSynchronize(stream);
                if res_sync != del_cudarc_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "stream sync failed",
                    ));
                }
                // del_cudarc_sys::cuStreamDestroy_v2(stream);
            }
        }
        _ => println!("Unknown device type {}", vtx2idx.ctx.device_type),
    }
    Ok(())
}

/// Solve the linear system from screened Poisson equation using Jacobi method:
///
/// {vtx2lhs} = L * {vtx2rhs}
/// where L = \[ .., -1, .., valence, ..,-1, .. \]
#[pyo3::pyfunction]
fn vtx2vtx_multiply_graph_laplacian(
    _py: Python,
    vtx2idx: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    vtx2lhs: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2idx = crate::get_managed_tensor_from_pyany(vtx2idx)?;
    let idx2vtx = crate::get_managed_tensor_from_pyany(idx2vtx)?;
    let vtx2rhs = crate::get_managed_tensor_from_pyany(vtx2rhs)?;
    let vtx2lhs = crate::get_managed_tensor_from_pyany(vtx2lhs)?;
    let vtx2idx_sh = unsafe { std::slice::from_raw_parts(vtx2idx.shape, vtx2idx.ndim as usize) };
    let vtx2lhs_sh = unsafe { std::slice::from_raw_parts(vtx2lhs.shape, vtx2lhs.ndim as usize) };
    let num_vtx = vtx2idx_sh[0] - 1;
    let num_dim = vtx2lhs_sh[1];
    match vtx2idx.ctx.device_type {
        dlpack::device_type_codes::CPU => {
            let (vtx2idx, vtx2idx_sh) =
                unsafe { crate::slice_shape_from_tensor::<u32>(vtx2idx).unwrap() };
            assert_eq!(vtx2idx_sh, vec!(num_vtx + 1));
            let (idx2vtx, idx2vtx_sh) =
                unsafe { crate::slice_shape_from_tensor::<u32>(idx2vtx).unwrap() };
            assert_eq!(idx2vtx_sh.len(), 1);
            let (vtx2rhs, vtx2rhs_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(vtx2rhs).unwrap() };
            assert_eq!(vtx2rhs_sh, vec!(num_vtx, num_dim));
            let (vtx2lhs, vtx2lhs_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<f32>(vtx2lhs).unwrap() };
            assert_eq!(vtx2lhs_sh, vec!(num_vtx, num_dim));
            del_msh_cpu::vtx2vtx::multiply_graph_laplacian::<3, u32>(
                vtx2idx, idx2vtx, vtx2rhs, vtx2lhs,
            );
            Ok(())
        }
        _ => {
            todo!()
        }
    }
}

#[pyfunction]
fn vtx2vtx_from_uniform_mesh(
    py: Python<'_>,
    elem2vtx: &Bound<'_, PyAny>,
    num_vtx: usize,
    is_self: bool,
) -> PyResult<(pyo3::PyObject, pyo3::PyObject)> {
    let elem2vtx = crate::get_managed_tensor_from_pyany(elem2vtx)?;
    match elem2vtx.ctx.device_type {
        dlpack::device_type_codes::CPU => {
            if elem2vtx.dtype.code == dlpack::data_type_codes::UINT && elem2vtx.dtype.bits == 64 {
                let (elem2vtx, elem2vtx_sh) =
                    unsafe { crate::slice_shape_from_tensor::<u64>(elem2vtx).unwrap() };
                let (vtx2idx, idx2vtx) = del_msh_cpu::vtx2vtx::from_uniform_mesh(
                    elem2vtx,
                    elem2vtx_sh[1] as usize,
                    num_vtx,
                    is_self,
                );
                let vtx2idx_cap =
                    crate::make_capsule_from_vec(py, vec![vtx2idx.len() as i64], vtx2idx);
                let idx2vtx_cap =
                    crate::make_capsule_from_vec(py, vec![idx2vtx.len() as i64], idx2vtx);
                return Ok((vtx2idx_cap, idx2vtx_cap));
            } else if elem2vtx.dtype.code == dlpack::data_type_codes::UINT
                && elem2vtx.dtype.bits == 32
            {
                let (elem2vtx, elem2vtx_sh) =
                    unsafe { crate::slice_shape_from_tensor::<u32>(elem2vtx).unwrap() };
                let (vtx2idx, idx2vtx) = del_msh_cpu::vtx2vtx::from_uniform_mesh(
                    elem2vtx,
                    elem2vtx_sh[1] as usize,
                    num_vtx,
                    is_self,
                );
                let vtx2idx_cap =
                    crate::make_capsule_from_vec(py, vec![vtx2idx.len() as i64], vtx2idx);
                let idx2vtx_cap =
                    crate::make_capsule_from_vec(py, vec![idx2vtx.len() as i64], idx2vtx);
                return Ok((vtx2idx_cap, idx2vtx_cap));
            }
        }
        _ => {
            todo!()
        }
    }
    todo!();
}
