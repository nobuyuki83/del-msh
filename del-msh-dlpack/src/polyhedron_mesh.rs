use del_dlpack::{
    check_1d_tensor as chk1, check_2d_tensor as chk2, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(polyhedron_mesh_elem2volume, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyhedron_mesh_elem2center, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        polyhedron_mesh_bvhnode2aabb_from_bvhnodes,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        polyhedron_mesh_search_elem_contain_points,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyhedron_mesh_subdivide, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        polyhedron_mesh_interpolate_values_at_points,
        m
    )?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_elem2volume(
    _py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    elem2volume: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let elem2volume = get_tensor(elem2volume)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    //
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk1::<f32>(elem2volume, num_elem, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::polyhedron_mesh::elem2volume(
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                1,
                slice_mut!(elem2volume, f32).unwrap(),
            );
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
fn polyhedron_mesh_elem2center(
    _py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    elem2center: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let elem2center = get_tensor(elem2center)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_dim = shape(vtx2xyz, 1).unwrap();
    //
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, num_dim, device).unwrap();
    chk2::<f32>(elem2center, num_elem, num_dim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let result = del_msh_cpu::elem2center::from_polygon_mesh_as_points(
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                num_dim.try_into().unwrap(),
            );
            slice_mut!(elem2center, f32)
                .unwrap()
                .copy_from_slice(&result);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::elem2center",
                del_msh_cuda_kernels::get("elem2center").unwrap(),
                "from_polygon_mesh_as_points",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_elem as u32);
            builder.arg_data(&elem2idx_offset.data);
            builder.arg_data(&idx2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_u32(num_dim as u32);
            builder.arg_data(&elem2center.data);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_elem as u32),
                )
                .unwrap();
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported (compile with --features cuda)",
            ));
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_bvhnode2aabb_from_bvhnodes(
    _py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    bvhnodes: &Bound<'_, PyAny>,
    bvhnode2aabb: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let bvhnodes = get_tensor(bvhnodes)?;
    let bvhnode2aabb = get_tensor(bvhnode2aabb)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_bvhnode = shape(bvhnodes, 0).unwrap();
    //
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<u32>(bvhnodes, num_bvhnode, 3, device).unwrap();
    chk2::<f32>(bvhnode2aabb, num_bvhnode, 6, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::bvhnode2aabb3::update_for_polygon_polyhedron_mesh_with_bvh::<u32, f32>(
                slice_mut!(bvhnode2aabb, f32).unwrap(),
                0,
                slice!(bvhnodes, u32).unwrap(),
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let num_branch = num_elem - 1;
            let bvhbranch2flag =
                del_cudarc_sys::CuVec::<u32>::alloc_zeros(num_branch as usize, stream).unwrap();
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::bvhnode2aabb",
                del_msh_cuda_kernels::get("bvhnode2aabb").unwrap(),
                "from_polyhedron_mesh",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_data(&bvhnode2aabb.data);
            builder.arg_dptr(bvhbranch2flag.dptr);
            builder.arg_u32(num_bvhnode as u32);
            builder.arg_data(&bvhnodes.data);
            builder.arg_u32(num_elem as u32);
            builder.arg_data(&elem2idx_offset.data);
            builder.arg_data(&idx2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_f32(0f32);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_elem as u32),
                )
                .unwrap();
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported (compile with --features cuda)",
            ));
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_search_elem_contain_points(
    _py: Python<'_>,
    bvhnodes: &Bound<'_, PyAny>,
    bvhnode2aabb: &Bound<'_, PyAny>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    wtx2xyz: &Bound<'_, PyAny>,
    wtx2elem: &Bound<'_, PyAny>,
    wtx2param: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let bvhnodes = get_tensor(bvhnodes)?;
    let bvhnode2aabb = get_tensor(bvhnode2aabb)?;
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let wtx2xyz = get_tensor(wtx2xyz)?;
    let wtx2elem = get_tensor(wtx2elem)?;
    let wtx2param = get_tensor(wtx2param)?;
    //
    let device = bvhnodes.ctx.device_type;
    let num_bvhnode = shape(bvhnodes, 0).unwrap();
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_wtx = shape(wtx2xyz, 0).unwrap();
    //
    chk2::<u32>(bvhnodes, num_bvhnode, 3, device).unwrap();
    chk2::<f32>(bvhnode2aabb, num_bvhnode, 6, device).unwrap();
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(wtx2xyz, num_wtx, 3, device).unwrap();
    chk1::<u32>(wtx2elem, num_wtx, device).unwrap();
    chk2::<f32>(wtx2param, num_wtx, 3, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let (res_elem, res_param) = del_msh_cpu::polyhedron_mesh::search_elem_contain_points(
                slice!(bvhnodes, u32).unwrap(),
                slice!(bvhnode2aabb, f32).unwrap(),
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice!(wtx2xyz, f32).unwrap(),
            );
            slice_mut!(wtx2elem, u32)
                .unwrap()
                .copy_from_slice(&res_elem);
            slice_mut!(wtx2param, f32)
                .unwrap()
                .copy_from_slice(&res_param);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

/// Subdivide a mixed polyhedron mesh once.
/// Returns three dlpack capsules: (elem2idx_offset, idx2vtx, vtx2xyz) for the refined mesh.
#[pyfunction]
fn polyhedron_mesh_subdivide(
    py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    //
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let (new_elem2idx, new_idx2vtx, new_vtx2xyz) = del_msh_cpu::polyhedron_mesh::subdivide(
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
            );
            let num_new_elem = new_elem2idx.len() - 1;
            let num_new_idx = new_idx2vtx.len();
            let num_new_vtx = new_vtx2xyz.len() / 3;
            let cap0 =
                del_dlpack::make_capsule_from_vec(py, vec![num_new_elem as i64 + 1], new_elem2idx);
            let cap1 = del_dlpack::make_capsule_from_vec(py, vec![num_new_idx as i64], new_idx2vtx);
            let cap2 =
                del_dlpack::make_capsule_from_vec(py, vec![num_new_vtx as i64, 3], new_vtx2xyz);
            Ok((cap0, cap1, cap2))
        }
        _ => {
            todo!()
        }
    }
}

/// Interpolate vertex values at query points using element shape functions.
/// `vtx2value` holds per-vertex data of shape `(num_vtx, num_value_dim)`.
/// `wtx2value` is filled with the interpolated values at each query point.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_interpolate_values_at_points(
    _py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2value: &Bound<'_, PyAny>,
    wtx2elem: &Bound<'_, PyAny>,
    wtx2param: &Bound<'_, PyAny>,
    wtx2value: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2value = get_tensor(vtx2value)?;
    let wtx2elem = get_tensor(wtx2elem)?;
    let wtx2param = get_tensor(wtx2param)?;
    let wtx2value = get_tensor(wtx2value)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2value, 0).unwrap();
    let num_value_dim = shape(vtx2value, 1).unwrap();
    let num_wtx = shape(wtx2elem, 0).unwrap();
    //
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2value, num_vtx, num_value_dim, device).unwrap();
    chk1::<u32>(wtx2elem, num_wtx, device).unwrap();
    chk2::<f32>(wtx2param, num_wtx, 3, device).unwrap();
    chk2::<f32>(wtx2value, num_wtx, num_value_dim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::polyhedron_mesh::interpolate_values_at_points(
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2value, f32).unwrap(),
                slice!(wtx2elem, u32).unwrap(),
                slice!(wtx2param, f32).unwrap(),
                num_value_dim as usize,
                slice_mut!(wtx2value, f32).unwrap(),
            );
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
