use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        edge2vtx_contour_for_triangle_mesh,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        edge2vtx_from_vtx2vtx,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn edge2vtx_contour_for_triangle_mesh(
    py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_world2ndc: &Bound<'_, PyAny>,
    edge2vtx: &Bound<'_, PyAny>,
    edge2tri: &Bound<'_, PyAny>,
) -> PyResult<pyo3::Py<PyAny>> {
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let transform_world2ndc = del_dlpack::get_managed_tensor_from_pyany(transform_world2ndc)?;
    let edge2vtx = del_dlpack::get_managed_tensor_from_pyany(edge2vtx)?;
    let edge2tri = del_dlpack::get_managed_tensor_from_pyany(edge2tri)?;
    //
    let num_tri = del_dlpack::get_shape_tensor(&tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(&vtx2xyz, 0).unwrap();
    let num_edge = del_dlpack::get_shape_tensor(&edge2vtx, 0).unwrap();
    let device = tri2vtx.ctx.device_type;
    //
    del_dlpack::check_2d_tensor::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(edge2vtx, num_edge, 2, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(edge2tri, num_edge, 2, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx).unwrap() };
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let transform_world2ndc =
                unsafe { del_dlpack::slice_from_tensor::<f32>(transform_world2ndc).unwrap() };
            let transform_world2ndc = arrayref::array_ref![transform_world2ndc, 0, 16];
            let edge2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(edge2vtx).unwrap() };
            let edge2tri = unsafe { del_dlpack::slice_from_tensor::<u32>(edge2tri).unwrap() };
            //
            let edge2vtx_contour = del_msh_cpu::edge2vtx::contour_for_triangle_mesh::<u32>(
                tri2vtx,
                vtx2xyz,
                transform_world2ndc,
                edge2vtx,
                edge2tri,
            );
            let num_contour = edge2vtx_contour.len() as i64 / 2;
            Ok(del_dlpack::make_capsule_from_vec(
                py,
                vec![num_contour, 2],
                edge2vtx_contour,
            ))
        }
        _ => {
            todo!();
        }
    }
}

#[pyo3::pyfunction]
pub fn edge2vtx_from_vtx2vtx(
    _py: Python<'_>,
    vtx2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    edge2vtx: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2idx_offset = del_dlpack::get_managed_tensor_from_pyany(vtx2idx_offset)?;
    let idx2vtx = del_dlpack::get_managed_tensor_from_pyany(idx2vtx)?;
    let edge2vtx = del_dlpack::get_managed_tensor_from_pyany(edge2vtx)?;
    //
    let num_vtx = del_dlpack::get_shape_tensor(&vtx2idx_offset, 0).unwrap() - 1;
    let num_idx = del_dlpack::get_shape_tensor(&idx2vtx, 0).unwrap();
    let num_edge = del_dlpack::get_shape_tensor(&edge2vtx, 0).unwrap();
    let device = vtx2idx_offset.ctx.device_type;
    //
    assert_eq!(num_edge, num_idx);
    del_dlpack::check_1d_tensor::<u32>(vtx2idx_offset, num_vtx + 1, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(idx2vtx, num_idx, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(edge2vtx, num_edge, 2, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let vtx2idx_offset = unsafe { del_dlpack::slice_from_tensor::<u32>(vtx2idx_offset).unwrap() };
            let idx2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(idx2vtx).unwrap() };
            let edge2vtx = unsafe { del_dlpack::slice_from_tensor_mut::<u32>(edge2vtx).unwrap() };
            del_msh_cpu::edge2vtx::from_vtx2vtx(vtx2idx_offset, idx2vtx, edge2vtx);
        },
        _ => {todo!()}
    }
    Ok(())
}

