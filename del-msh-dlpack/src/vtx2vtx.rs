use del_dlpack::{
    check_1d_tensor as chk1, check_2d_tensor as chk2, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape,
    make_capsule_from_vec as capsule, pyo3, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

#[cfg(feature = "cuda")]
use del_cudarc_sys::{cu, CuVec, LaunchConfig};
#[cfg(feature = "cuda")]
use del_dlpack::del_cudarc_sys;

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(vtx2vtx_laplacian_smoothing, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vtx2vtx_graph_screened_poisson, m)?)?;
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
fn vtx2vtx_graph_screened_poisson(
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
            del_msh_cpu::vtx2vtx::graph_screend_poisson::<u32>(
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
            //let function = crate::load_get_function("vtx2vtx", "laplacian_smoothing").unwrap();
            let function = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh__vtx2vtx",
                del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
                "graph_screened_poisson",
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
                            del_cudarc_sys::LaunchConfig::from_num_thread_for_max_occupancy(
                                num_vtx as u32,
                                function,
                                0,
                                0,
                            )
                            .unwrap(),
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
                            del_cudarc_sys::LaunchConfig::from_num_thread_for_max_occupancy(
                                num_vtx as u32,
                                function,
                                0,
                                0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                }
            }
        }
        _ => println!("Unknown device type {}", vtx2idx_offset.ctx.device_type),
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
fn vtx2vtx_laplacian_smoothing(
    _py: Python,
    vtx2idx: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    lambda: f32,
    vtx2val: &Bound<'_, PyAny>,
    vtx2ave: &Bound<'_, PyAny>,
    num_iter: usize,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let vtx2idx_offset = get_tensor(vtx2idx)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2val = get_tensor(vtx2val)?;
    let vtx2ave = get_tensor(vtx2ave)?;
    //
    let num_vtx = shape(vtx2idx_offset, 0).unwrap() - 1;
    let num_vdim = shape(vtx2val, 1).unwrap();
    let device_type = vtx2idx_offset.ctx.device_type;
    //
    chk1::<u32>(vtx2idx_offset, num_vtx + 1, device_type).unwrap();
    chk1::<u32>(idx2vtx, -1, device_type).unwrap();
    chk2::<f32>(vtx2val, num_vtx, num_vdim, device_type).unwrap();
    chk2::<f32>(vtx2ave, num_vtx, num_vdim, device_type).unwrap();
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::vtx2vtx::laplacian_smoothing::<u32>(
                slice!(vtx2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                lambda,
                num_vdim as usize,
                slice_mut!(vtx2val, f32).unwrap(),
                slice_mut!(vtx2ave, f32).unwrap(),
                num_iter,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
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
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_u32(num_vtx as u32);
                builder.arg_data(&vtx2idx_offset.data);
                builder.arg_data(&idx2vtx.data);
                builder.arg_f32(lambda);
                builder.arg_u32(num_vdim as u32);
                builder.arg_data(&vtx2val.data);
                builder.arg_data(&vtx2ave.data);
                builder
                    .launch_kernel(
                        function,
                        del_cudarc_sys::LaunchConfig::from_num_thread_for_max_occupancy(
                            num_vtx as u32,
                            function,
                            0,
                            0,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                {
                    del_cudarc_sys::memcpy_d2d_32(
                        vtx2val.data as del_cudarc_sys::cu::CUdeviceptr,
                        vtx2ave.data as del_cudarc_sys::cu::CUdeviceptr,
                        (num_vtx * num_vdim) as usize,
                        stream,
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
            multiply_graph_laplacian(
                stream,
                &vtx2idx,
                &idx2vtx,
                num_vdim as usize,
                &vtx2rhs,
                &vtx2lhs,
            );
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported (compile with --features cuda)",
            ))
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
            let s = slice!(elem2vtx, u32).unwrap();
            let (vtx2idx, idx2vtx) = match num_node as usize {
                2 => {
                    del_msh_cpu::vtx2vtx::from_uniform_mesh(s.as_chunks::<2>().0, num_vtx, is_self)
                }
                3 => {
                    del_msh_cpu::vtx2vtx::from_uniform_mesh(s.as_chunks::<3>().0, num_vtx, is_self)
                }
                4 => {
                    del_msh_cpu::vtx2vtx::from_uniform_mesh(s.as_chunks::<4>().0, num_vtx, is_self)
                }
                _ => panic!("unsupported num_node: {num_node}"),
            };
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
            let (vtx2idx, idx2vtx) =
                from_uniform_mesh(stream, &elem2vtx, num_elem as usize, num_vtx, is_self);
            let vtx2idx_cap =
                del_dlpack::make_capsule_from_cuvec(py, 0, vec![vtx2idx.n as i64], vtx2idx);
            let idx2vtx_cap =
                del_dlpack::make_capsule_from_cuvec(py, 0, vec![idx2vtx.n as i64], idx2vtx);
            Ok((vtx2idx_cap, idx2vtx_cap))
        }
        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GPU not supported (compile with --features cuda)",
        )),
    }
}

// ------------------------------

#[cfg(feature = "cuda")]
pub fn from_uniform_mesh(
    stream: cu::CUstream,
    elem2vtx: &CuVec<u32>,
    num_elem: usize,
    num_vtx: usize,
    is_self: bool,
) -> (CuVec<u32>, CuVec<u32>) {
    let num_node = elem2vtx.n / num_elem;
    assert_eq!(elem2vtx.n, num_elem * num_node);
    let (vtx2jdx, jdx2elem) =
        crate::vtx2elem::from_uniform_mesh(stream, elem2vtx, num_elem, num_vtx);
    let num_buff = if is_self {
        jdx2elem.n * num_node
    } else {
        jdx2elem.n * (num_node - 1)
    };
    let jdx2buff = CuVec::<u32>::with_capacity(num_buff).unwrap();
    let vtx2nvtx = CuVec::<u32>::with_capacity(num_vtx + 1).unwrap();
    vtx2nvtx.set_zeros(stream).unwrap();
    {
        /*
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::VTX2VTX,
            "vtx2nvtx_from_uniform_mesh",
        )
        .unwrap();
         */
        // let func = crate::load_get_function("vtx2vtx", "vtx2nvtx_from_uniform_mesh").unwrap();
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_msh::vtx2vtx",
            del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
            "vtx2nvtx_from_uniform_mesh",
        )
        .unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_vtx as u32);
        builder.arg_dptr(elem2vtx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_bool(is_self);
        builder.arg_dptr(vtx2jdx.dptr);
        builder.arg_dptr(jdx2elem.dptr);
        builder.arg_dptr(vtx2nvtx.dptr);
        builder.arg_dptr(jdx2buff.dptr);
        builder
            .launch_kernel(
                func,
                LaunchConfig::from_num_thread_for_max_occupancy(num_vtx as u32, func, 0, 0)
                    .unwrap(),
            )
            .unwrap();
    }
    let vtx2idx = CuVec::<u32>::with_capacity(vtx2nvtx.n).unwrap();
    del_cudarc_sys::cumsum::exclusive_scan(stream, &vtx2nvtx, &vtx2idx);
    let num_idx = vtx2idx.last().unwrap();
    let idx2vtx: CuVec<u32> = CuVec::with_capacity(num_idx as usize).unwrap();
    {
        /*
        let (func, _mdl) = del_cudarc_sys::load_function_in_module(
            del_msh_cuda_kernel::VTX2VTX,
            "idx2vtx_from_vtx2buff_for_uniform_mesh",
        )
        .unwrap();
         */
        /*
        let func =
            crate::load_get_function("vtx2vtx", "idx2vtx_from_vtx2buff_for_uniform_mesh").unwrap();
         */
        let func = del_cudarc_sys::cache_func::get_function_cached(
            "del_msh::vtx2vtx",
            del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
            "idx2vtx_from_vtx2buff_for_uniform_mesh",
        )
        .unwrap();
        let mut builder = del_cudarc_sys::Builder::new(stream);
        builder.arg_u32(num_vtx as u32);
        builder.arg_dptr(vtx2jdx.dptr);
        builder.arg_u32(num_node as u32);
        builder.arg_bool(is_self);
        builder.arg_dptr(vtx2idx.dptr);
        builder.arg_dptr(jdx2buff.dptr);
        builder.arg_dptr(idx2vtx.dptr);
        builder
            .launch_kernel(
                func,
                LaunchConfig::from_num_thread_for_max_occupancy(num_vtx as u32, func, 0, 0)
                    .unwrap(),
            )
            .unwrap();
    }
    (vtx2idx, idx2vtx)
}

#[cfg(feature = "cuda")]
pub fn multiply_graph_laplacian(
    stream: cu::CUstream,
    vtx2idx_offset: &CuVec<u32>,
    idx2vtx: &CuVec<u32>,
    num_vdim: usize,
    vtx2rhs: &CuVec<f32>,
    vtx2lhs: &CuVec<f32>,
) {
    let num_vtx = vtx2idx_offset.n - 1;
    assert_eq!(vtx2rhs.n, num_vtx * num_vdim);
    assert_eq!(vtx2lhs.n, num_vtx * num_vdim);
    /*
    let (func, _mdl) = del_cudarc_sys::load_function_in_module(
        del_msh_cuda_kernel::VTX2VTX,
        "multiply_graph_laplacian",
    )
    .unwrap();
     */
    // let func = crate::load_get_function("vtx2vtx", "multiply_graph_laplacian").unwrap();
    let func = del_cudarc_sys::cache_func::get_function_cached(
        "del_msh::vtx2vtx",
        del_msh_cuda_kernels::get("vtx2vtx").unwrap(),
        "multiply_graph_laplacian",
    )
    .unwrap();
    let mut builder = del_cudarc_sys::Builder::new(stream);
    builder.arg_u32(num_vtx as u32);
    builder.arg_dptr(vtx2idx_offset.dptr);
    builder.arg_dptr(idx2vtx.dptr);
    builder.arg_u32(num_vdim as u32);
    builder.arg_dptr(vtx2rhs.dptr);
    builder.arg_dptr(vtx2lhs.dptr);
    builder
        .launch_kernel(
            func,
            LaunchConfig::from_num_thread_for_max_occupancy(num_vtx as u32, func, 0, 0).unwrap(),
        )
        .unwrap();
}
