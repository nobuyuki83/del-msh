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
        polyhedron_mesh_nearest_elem_for_points,
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
        _ => {
            todo!()
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
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_nearest_elem_for_points(
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
            let (res_elem, res_param) = del_msh_cpu::polyhedron_mesh::nearest_elem_for_points(
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
