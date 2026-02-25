use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_raycast_update_pix2tri, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn trimesh3_raycast_update_pix2tri(
    _py: Python<'_>,
    pix2tri: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    bvhnodes: &Bound<'_, PyAny>,
    bvhnode2aabb: &Bound<'_, PyAny>,
    transform_ndc2world: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let pix2tri = del_dlpack::get_managed_tensor_from_pyany(pix2tri)?;
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let bvhnodes = del_dlpack::get_managed_tensor_from_pyany(bvhnodes)?;
    let bvhnode2aabb = del_dlpack::get_managed_tensor_from_pyany(bvhnode2aabb)?;
    let transform_ndc2world = del_dlpack::get_managed_tensor_from_pyany(transform_ndc2world)?;
    //
    let device = pix2tri.ctx.device_type;
    let pix2tri_sh = unsafe { std::slice::from_raw_parts(pix2tri.shape, pix2tri.ndim as usize) };
    let img_shape = [
        del_dlpack::get_shape_tensor(pix2tri, 1).unwrap(),
        del_dlpack::get_shape_tensor(pix2tri, 0).unwrap(),
    ];
    let num_tri = del_dlpack::get_shape_tensor(tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    let num_bvhnode = del_dlpack::get_shape_tensor(bvhnodes, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<u32>(pix2tri, img_shape[1], img_shape[0], device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(bvhnodes, num_bvhnode, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(bvhnode2aabb, num_bvhnode, 6, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let pix2tri = unsafe { del_dlpack::slice_from_tensor_mut::<u32>(pix2tri).unwrap() };
            let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let bvhnodes = unsafe { del_dlpack::slice_from_tensor::<u32>(bvhnodes).unwrap() };
            let bvhnode2aabb =
                unsafe { del_dlpack::slice_from_tensor::<f32>(bvhnode2aabb).unwrap() };
            let transform_ndc2world =
                unsafe { del_dlpack::slice_from_tensor::<f32>(transform_ndc2world).unwrap() };
            let transform_ndc2world = arrayref::array_ref![transform_ndc2world, 0, 16];
            let img_shape = (pix2tri_sh[1] as usize, pix2tri_sh[0] as usize);
            del_msh_cpu::trimesh3_raycast::update_pix2tri(
                pix2tri,
                tri2vtx,
                vtx2xyz,
                bvhnodes,
                bvhnode2aabb,
                img_shape,
                transform_ndc2world,
            );
            Ok(())
        }
        _ => {
            todo!()
        }
    }
}
