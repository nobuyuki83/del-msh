use crate::{check_1d_tensor, check_2d_tensor, get_shape_tensor};
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
    let num_vtx = get_shape_tensor(vtx2idx, 0) - 1;
    let num_dim = get_shape_tensor(vtx2rhs, 1);
    let device_type = vtx2idx.ctx.device_type;
    //
    check_1d_tensor::<u32>(vtx2idx, num_vtx + 1, device_type);
    check_1d_tensor::<u32>(idx2vtx, -1, device_type);
    check_2d_tensor::<f32>(vtx2rhs, num_vtx, num_dim, device_type);
    check_2d_tensor::<f32>(vtx2lhs, num_vtx, num_dim, device_type);
    check_2d_tensor::<f32>(vtx2lhstmp, num_vtx, num_dim, device_type);
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            let vtx2idx = unsafe { crate::slice_from_tensor::<u32>(vtx2idx).unwrap() };
            let idx2vtx = unsafe { crate::slice_from_tensor::<u32>(idx2vtx).unwrap() };
            let vtx2rhs = unsafe { crate::slice_from_tensor::<f32>(vtx2rhs).unwrap() };
            let vtx2lhs = unsafe { crate::slice_from_tensor_mut::<f32>(vtx2lhs).unwrap() };
            let vtx2lhstmp = unsafe { crate::slice_from_tensor_mut::<f32>(vtx2lhstmp).unwrap() };
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
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
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
                    builder.launch_kernel(
                        function,
                        del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32),
                    );
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
                    builder.launch_kernel(
                        function,
                        del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32),
                    );
                }
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
    //
    let num_vtx = get_shape_tensor(vtx2idx, 0) - 1;
    let num_dim = get_shape_tensor(vtx2lhs, 1);
    let device = vtx2idx.ctx.device_type;
    //
    assert_eq!(num_dim, 3);
    check_1d_tensor::<u32>(vtx2idx, num_vtx + 1, device);
    check_1d_tensor::<u32>(idx2vtx, -1, device);
    check_2d_tensor::<f32>(vtx2rhs, num_vtx, 3, device);
    check_2d_tensor::<f32>(vtx2lhs, num_vtx, 3, device);
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2idx = unsafe { crate::slice_from_tensor::<u32>(vtx2idx).unwrap() };
            let idx2vtx = unsafe { crate::slice_from_tensor::<u32>(idx2vtx).unwrap() };
            let vtx2rhs = unsafe { crate::slice_from_tensor::<f32>(vtx2rhs).unwrap() };
            let vtx2lhs = unsafe { crate::slice_from_tensor_mut::<f32>(vtx2lhs).unwrap() };
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
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let elem2vtx = crate::get_managed_tensor_from_pyany(elem2vtx)?;
    //
    let num_elem = get_shape_tensor(elem2vtx, 0);
    let num_node = get_shape_tensor(elem2vtx, 1);
    let device = elem2vtx.ctx.device_type;
    check_2d_tensor::<u32>(elem2vtx, num_elem, num_node, device);
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let elem2vtx = unsafe { crate::slice_from_tensor::<u32>(elem2vtx).unwrap() };
            let (vtx2idx, idx2vtx) = del_msh_cpu::vtx2vtx::from_uniform_mesh(
                elem2vtx,
                num_node as usize,
                num_vtx,
                is_self,
            );
            let vtx2idx_cap = crate::make_capsule_from_vec(py, vec![vtx2idx.len() as i64], vtx2idx);
            let idx2vtx_cap = crate::make_capsule_from_vec(py, vec![idx2vtx.len() as i64], idx2vtx);
            Ok((vtx2idx_cap, idx2vtx_cap))
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            use del_cudarc_sys::cu::CUdeviceptr;
            let elem2vtx = CuVec::new(
                elem2vtx.data as CUdeviceptr,
                (num_elem * num_node) as usize,
                false,
            );
            let (vtx2idx, idx2vtx) = del_msh_cudarc_sys::vtx2vtx::from_uniform_mesh(
                stream,
                &elem2vtx,
                num_elem as usize,
                num_vtx,
                is_self,
            );
            let vtx2idx_cap =
                crate::make_capsule_from_cuvec(py, 0, vec![vtx2idx.n as i64], vtx2idx);
            let idx2vtx_cap =
                crate::make_capsule_from_cuvec(py, 0, vec![idx2vtx.n as i64], idx2vtx);
            Ok((vtx2idx_cap, idx2vtx_cap))
        }
        _ => {
            todo!()
        }
    }
}
