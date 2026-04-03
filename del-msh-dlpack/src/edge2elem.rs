use del_dlpack::dlpack;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        edge2elem_from_edge2vtx_of_tri2vtx_with_vtx2vtx,
        m
    )?)?;
    Ok(())
}



#[pyo3::pyfunction]
pub fn edge2elem_from_edge2vtx_of_tri2vtx_with_vtx2vtx(
    _py: Python<'_>,
    edge2vtx: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    edge2tri: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let edge2vtx = del_dlpack::get_managed_tensor_from_pyany(edge2vtx)?;
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2idx_offset = del_dlpack::get_managed_tensor_from_pyany(vtx2idx_offset)?;
    let idx2vtx = del_dlpack::get_managed_tensor_from_pyany(idx2vtx)?;
    let edge2tri = del_dlpack::get_managed_tensor_from_pyany(edge2tri)?;
    //
    let device = edge2vtx.ctx.device_type;
    let num_edge = del_dlpack::get_shape_tensor(&edge2vtx, 0).unwrap();
    let num_tri = del_dlpack::get_shape_tensor(&tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(&vtx2idx_offset, 0).unwrap() - 1;
    let num_idx = del_dlpack::get_shape_tensor(&idx2vtx, 0).unwrap();
    //
    assert_eq!(num_edge, num_idx);
    del_dlpack::check_2d_tensor::<u32>(&edge2vtx, num_edge, 2, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(&tri2vtx, num_tri, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(&vtx2idx_offset, num_vtx+1, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(&idx2vtx, num_idx, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(&edge2tri, num_edge, 2, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let edge2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(edge2vtx).unwrap() };
            let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx).unwrap() };
            let vtx2idx_offset = unsafe { del_dlpack::slice_from_tensor::<u32>(vtx2idx_offset).unwrap() };
            let idx2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(idx2vtx).unwrap() };
            let edge2tri = unsafe { del_dlpack::slice_from_tensor_mut::<u32>(edge2tri).unwrap() };
            del_msh_cpu::edge2elem::from_edge2vtx_of_tri2vtx_with_vtx2vtx(
                edge2vtx,
                tri2vtx,
                vtx2idx_offset,
                idx2vtx,
                edge2tri
            );
        },
        _ => {}
    }
    Ok(())
}