use numpy::{PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::{types::PyModule, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_raycast_update_pix2tri, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn trimesh3_raycast_update_pix2tri<'a>(
    _py: Python<'a>,
    mut pix2tri: PyReadwriteArray2<'a, usize>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
    bvhnodes: PyReadonlyArray2<'a, usize>,
    bvhnode2aabb: PyReadonlyArray2<'a, f32>,
    transform_ndc2world: PyReadwriteArray1<'a, f32>,
) {
    assert!(pix2tri.is_c_contiguous());
    //
    assert!(tri2vtx.is_c_contiguous());
    assert_eq!(tri2vtx.shape()[1], 3);
    let num_tri = tri2vtx.shape()[0];
    //
    assert!(vtx2xyz.is_c_contiguous());
    assert_eq!(vtx2xyz.shape()[1], 3);
    //
    assert!(bvhnodes.is_c_contiguous());
    assert_eq!(bvhnodes.shape()[0], 2 * num_tri - 1);
    assert_eq!(bvhnodes.shape()[1], 3);
    //
    assert!(bvhnode2aabb.is_c_contiguous());
    assert_eq!(bvhnode2aabb.shape()[1], 6);
    use numpy::PyUntypedArrayMethods;
    let img_shape = (pix2tri.shape()[1], pix2tri.shape()[0]);
    let transform_ndc2world = arrayref::array_ref![transform_ndc2world.as_slice().unwrap(), 0, 16];
    del_msh_cpu::trimesh3_raycast::update_pix2tri(
        pix2tri.as_slice_mut().unwrap(),
        tri2vtx.as_slice().unwrap(),
        vtx2xyz.as_slice().unwrap(),
        bvhnodes.as_slice().unwrap(),
        bvhnode2aabb.as_slice().unwrap(),
        img_shape,
        transform_ndc2world,
    );
}
