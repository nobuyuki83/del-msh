use del_dlpack::{
    dlpack, pyo3,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape,
    check_1d_tensor as chk1, check_2d_tensor as chk2,
    make_capsule_from_vec as capsule, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

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
    let vtx2idx_offset = get_tensor(vtx2idx)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2rhs = get_tensor(vtx2rhs)?;
    let vtx2lhs = get_tensor(vtx2lhs)?;
    let vtx2lhstmp = get_tensor(vtx2lhstmp)?;
    //
    let num_vtx = shape(vtx2idx_offset, 0).unwrap() - 1;
    let num_dim = shape(vtx2rhs, 1).unwrap();
    let device_type = vtx2idx_offset.ctx.device_type;
    //
    chk1::<u32>(vtx2idx_offset, num_vtx + 1, device_type).unwrap();
    chk1::<u32>(idx2vtx, -1, device_type).unwrap();
    chk2::<f32>(vtx2rhs, num_vtx, num_dim, device_type).unwrap();
    chk2::<f32>(vtx2lhs, num_vtx, num_dim, device_type).unwrap();
    chk2::<f32>(vtx2lhstmp, num_vtx, num_dim, device_type).unwrap();
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::vtx2vtx::laplacian_smoothing::<u32>(
                slice!(vtx2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                lambda,
                3,
                slice_mut!(vtx2lhs, f32).unwrap(),
                slice!(vtx2rhs, f32).unwrap(),
                num_iter,
                slice_mut!(vtx2lhstmp, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            // println!("GPU_{}", vtx2idx.ctx.device_id);
            /*
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::VTX2VTX,
                "laplacian_smoothing",
            )
            .unwrap();
             */
            //let function = crate::load_get_function("vtx2vtx", "laplacian_smoothing").unwrap();
            let function = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__vtx2vtx",
                del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
                "laplacian_smoothing",
            )
            .unwrap();
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            for _itr in 0..num_iter {
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_u32(num_vtx as u32);
                    builder.arg_data(&vtx2idx_offset.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2lhstmp.data);
                    builder.arg_data(&vtx2lhs.data);
                    builder.arg_data(&vtx2rhs.data);
                    builder
                        .launch_kernel(
                            function,
                            del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32),
                        )
                        .unwrap();
                }
                {
                    let mut builder = del_cudarc_sys::Builder::new(stream);
                    builder.arg_u32(num_vtx as u32);
                    builder.arg_data(&vtx2idx_offset.data);
                    builder.arg_data(&idx2vtx.data);
                    builder.arg_f32(lambda);
                    builder.arg_data(&vtx2lhs.data);
                    builder.arg_data(&vtx2lhstmp.data);
                    builder.arg_data(&vtx2rhs.data);
                    builder
                        .launch_kernel(
                            function,
                            del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32),
                        )
                        .unwrap();
                }
            }
        }
        _ => println!("Unknown device type {}", vtx2idx_offset.ctx.device_type),
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
    vtx2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    vtx2lhs: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2idx = get_tensor(vtx2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2rhs = get_tensor(vtx2rhs)?;
    let vtx2lhs = get_tensor(vtx2lhs)?;
    //
    let num_vtx = shape(vtx2idx, 0).unwrap() - 1;
    let num_vdim = shape(vtx2lhs, 1).unwrap();
    let num_idx = shape(idx2vtx, 0).unwrap();
    let device = vtx2idx.ctx.device_type;
    //
    chk1::<u32>(vtx2idx, num_vtx + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2rhs, num_vtx, num_vdim, device).unwrap();
    chk2::<f32>(vtx2lhs, num_vtx, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::vtx2vtx::multiply_graph_laplacian::<u32>(
                slice!(vtx2idx, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                num_vdim as usize,
                slice!(vtx2rhs, f32).unwrap(),
                slice_mut!(vtx2lhs, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cu::CUdeviceptr, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let vtx2idx =
                CuVec::<u32>::from_dptr(vtx2idx.data as CUdeviceptr, num_vtx as usize + 1);
            let idx2vtx =
                CuVec::<u32>::from_dptr(idx2vtx.data as CUdeviceptr, num_idx as usize + 1);
            let vtx2rhs =
                CuVec::<f32>::from_dptr(vtx2rhs.data as CUdeviceptr, (num_vtx * num_vdim) as usize);
            let vtx2lhs =
                CuVec::<f32>::from_dptr(vtx2lhs.data as CUdeviceptr, (num_vtx * num_vdim) as usize);
            del_msh_cudarc_sys::vtx2vtx::multiply_graph_laplacian(
                stream,
                &vtx2idx,
                &idx2vtx,
                num_vdim as usize,
                &vtx2rhs,
                &vtx2lhs,
            );
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[del_dlpack::pyo3::pyfunction]
fn vtx2vtx_from_uniform_mesh(
    py: Python<'_>,
    elem2vtx: &Bound<'_, PyAny>,
    num_vtx: usize,
    is_self: bool,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let elem2vtx = get_tensor(elem2vtx)?;
    //
    let num_elem = shape(elem2vtx, 0).unwrap();
    let num_node = shape(elem2vtx, 1).unwrap();
    let device = elem2vtx.ctx.device_type;
    chk2::<u32>(elem2vtx, num_elem, num_node, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let (vtx2idx, idx2vtx) = del_msh_cpu::vtx2vtx::from_uniform_mesh(
                slice!(elem2vtx, u32).unwrap(),
                num_node as usize,
                num_vtx,
                is_self,
            );
            Ok((
                capsule(py, vec![vtx2idx.len() as i64], vtx2idx),
                capsule(py, vec![idx2vtx.len() as i64], idx2vtx),
            ))
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0)).unwrap();
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
                del_dlpack::make_capsule_from_cuvec(py, 0, vec![vtx2idx.n as i64], vtx2idx);
            let idx2vtx_cap =
                del_dlpack::make_capsule_from_cuvec(py, 0, vec![idx2vtx.n as i64], idx2vtx);
            Ok((vtx2idx_cap, idx2vtx_cap))
        }
        _ => {
            todo!()
        }
    }
}
