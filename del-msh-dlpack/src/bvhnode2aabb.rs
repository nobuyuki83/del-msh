use pyo3::{Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(bvhnode2aabb_update_aabb, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn bvhnode2aabb_update_aabb(
    _py: Python<'_>,
    bvhnode2aabb: &Bound<'_, PyAny>,
    i_bvhnode: usize,
    bvhnodes: &Bound<'_, PyAny>,
    elem2vtx: &Bound<'_, PyAny>,
    vtx2xyz0: &Bound<'_, PyAny>,
    vtx2xyz1: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let bvhnode2aabb = del_dlpack::get_managed_tensor_from_pyany(bvhnode2aabb)?;
    let bvhnodes = del_dlpack::get_managed_tensor_from_pyany(bvhnodes)?;
    let elem2vtx = del_dlpack::get_managed_tensor_from_pyany(elem2vtx)?;
    let vtx2xyz0 = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz0)?;
    let vtx2xyz1 = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz1)?;
    let device = bvhnode2aabb.ctx.device_type;
    let ndim = del_dlpack::get_shape_tensor(vtx2xyz0, 1).unwrap();
    let num_bvhnode = del_dlpack::get_shape_tensor(bvhnode2aabb, 0).unwrap();
    del_dlpack::check_2d_tensor::<f32>(bvhnode2aabb, num_bvhnode, ndim * 2, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(bvhnodes, num_bvhnode, 3, device).unwrap();
    let num_elem = (num_bvhnode + 1) / 2;
    match device {
        del_dlpack::dlpack::device_type_codes::CPU => match ndim {
            3 => {
                let bvhnode2aabb =
                    unsafe { del_dlpack::slice_from_tensor_mut(bvhnode2aabb) }.unwrap();
                let bvhnodes = unsafe { del_dlpack::slice_from_tensor(bvhnodes) }.unwrap();
                let vtx2xyz0 = unsafe { del_dlpack::slice_from_tensor(vtx2xyz0) }.unwrap();
                let vtx2xyz1 = unsafe { del_dlpack::slice_from_tensor(vtx2xyz1) }.unwrap();
                let vtx2xyz1 = if vtx2xyz0.len() == vtx2xyz1.len() {
                    Some(vtx2xyz1)
                } else {
                    None
                };
                let elem2vtx = unsafe { del_dlpack::slice_from_tensor(elem2vtx) };
                let elem2vtx =
                    elem2vtx.map(|elem2vtx| (elem2vtx, elem2vtx.len() / num_elem as usize));
                del_msh_cpu::bvhnode2aabb3::update_for_uniform_mesh_with_bvh::<u32, f32>(
                    bvhnode2aabb,
                    i_bvhnode,
                    bvhnodes,
                    elem2vtx,
                    vtx2xyz0,
                    vtx2xyz1,
                );
            }
            _ => {
                todo!()
            }
        },
        _ => {
            todo!()
        }
    }
    Ok(())
}
