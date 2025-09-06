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
    let pix2tri = crate::get_managed_tensor_from_pyany(pix2tri)?;
    let tri2vtx = crate::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = crate::get_managed_tensor_from_pyany(vtx2xyz)?;
    let bvhnodes = crate::get_managed_tensor_from_pyany(bvhnodes)?;
    let bvhnode2aabb = crate::get_managed_tensor_from_pyany(bvhnode2aabb)?;
    let transform_ndc2world = crate::get_managed_tensor_from_pyany(transform_ndc2world)?;
    match pix2tri.ctx.device_type {
        dlpack::device_type_codes::CPU => {
            assert_eq!(tri2vtx.ctx.device_type, dlpack::device_type_codes::CPU);
            let (pix2tri, pix2tri_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<usize>(pix2tri).unwrap() };
            assert_eq!(pix2tri_sh.len(), 2);
            //
            let (tri2vtx, tri2vtx_sh) =
                unsafe { crate::slice_shape_from_tensor::<usize>(tri2vtx).unwrap() };
            let num_tri = tri2vtx_sh[0];
            assert!(matches!(tri2vtx_sh[..], [_, 3]));
            //
            let (vtx2xyz, vtx2xyz_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(vtx2xyz).unwrap() };
            assert!(matches!(vtx2xyz_sh[..], [_, 3]));
            //
            let (bvhnodes, bvhnodes_sh) =
                unsafe { crate::slice_shape_from_tensor::<usize>(bvhnodes).unwrap() };
            assert_eq!(bvhnodes_sh, [num_tri * 2 - 1, 3]);
            let num_bvhnode = bvhnodes_sh[0];
            //
            let (bvhnode2aabb, bvhnode2aabb_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(bvhnode2aabb).unwrap() };
            assert_eq!(bvhnode2aabb_sh, [num_bvhnode, 6]);
            //
            let (transform_ndc2world, transform_ndc2world_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(transform_ndc2world).unwrap() };
            assert_eq!(transform_ndc2world_sh, [16]);
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
