use del_msh_cpu::vtx2xyz::transform_homogeneous;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        edge2vtx_contour_for_triangle_mesh,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn edge2vtx_contour_for_triangle_mesh<'a>(
    _py: Python<'a>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_world2ndc: &Bound<'_, PyAny>,
    edge2vtx: &Bound<'_, PyAny>,
    edge2tri: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tri2vtx = crate::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = crate::get_managed_tensor_from_pyany(vtx2xyz)?;
    let transform_world2ndc = crate::get_managed_tensor_from_pyany(transform_world2ndc)?;
    let edge2vtx = crate::get_managed_tensor_from_pyany(edge2vtx)?;
    let edge2tri = crate::get_managed_tensor_from_pyany(edge2tri)?;
    match tri2vtx.ctx.device_type {
        dlpack::device_type_codes::CPU => {
            let (tri2vtx, tri2vtx_sh) =
                unsafe { crate::slice_shape_from_tensor::<usize>(tri2vtx).unwrap() };
            //del_msh_cpu::edge2vtx::contour_for_triangle_mesh(tri2vtx, vtx2xyz, transform_world2ndc, edge2vtx, edge2tri);
            Ok(())
        }
        _ => {
            todo!();
        }
    }
}
