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
    let tri2vtx_sh = unsafe { std::slice::from_raw_parts(tri2vtx.shape, tri2vtx.ndim as usize) };
    let vtx2xyz_sh = unsafe { std::slice::from_raw_parts(vtx2xyz.shape, vtx2xyz.ndim as usize) };
    let bvhnodes_sh = unsafe { std::slice::from_raw_parts(bvhnodes.shape, bvhnodes.ndim as usize) };
    let bvhnode2aabb_sh =
        unsafe { std::slice::from_raw_parts(bvhnode2aabb.shape, bvhnode2aabb.ndim as usize) };
    let transform_ndc2world_sh = unsafe {
        std::slice::from_raw_parts(transform_ndc2world.shape, transform_ndc2world.ndim as usize)
    };
    let num_vtx = vtx2xyz_sh[0];
    let num_tri = tri2vtx_sh[0];
    let num_bvhnode = bvhnodes_sh[0];
    //
    assert_eq!(vtx2xyz_sh, vec!(num_vtx, 3));
    assert_eq!(tri2vtx_sh, vec!(num_tri, 3));
    assert_eq!(num_tri * 2 - 1, num_bvhnode);
    assert_eq!(bvhnodes_sh, vec![num_bvhnode, 3]);
    assert_eq!(bvhnode2aabb_sh, vec![num_bvhnode, 6]);
    assert_eq!(transform_ndc2world_sh, vec![16]);
    assert!(del_dlpack::is_equal::<usize>(&pix2tri.dtype));
    assert!(del_dlpack::is_equal::<usize>(&tri2vtx.dtype));
    assert!(del_dlpack::is_equal::<f32>(&vtx2xyz.dtype));
    assert!(del_dlpack::is_equal::<usize>(&bvhnodes.dtype));
    assert!(del_dlpack::is_equal::<f32>(&bvhnode2aabb.dtype));
    assert!(del_dlpack::is_equal::<f32>(&transform_ndc2world.dtype));
    assert_eq!(tri2vtx.ctx.device_type, device);
    assert_eq!(vtx2xyz.ctx.device_type, device);
    assert!(unsafe { del_dlpack::is_tensor_c_contiguous(vtx2xyz) });
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let pix2tri = unsafe { del_dlpack::slice_from_tensor_mut::<usize>(pix2tri).unwrap() };
            let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<usize>(tri2vtx).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let bvhnodes = unsafe { del_dlpack::slice_from_tensor::<usize>(bvhnodes).unwrap() };
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
