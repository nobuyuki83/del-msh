use del_dlpack::dlpack;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(mix_mesh_to_polyhedron_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn mix_mesh_to_polyhedron_mesh(
    _py: Python<'_>,
    tet2vtx: &Bound<'_, PyAny>,
    pyrmd2vtx: &Bound<'_, PyAny>,
    prism2vtx: &Bound<'_, PyAny>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tet2vtx = del_dlpack::get_managed_tensor_from_pyany(tet2vtx)?;
    let pyrmd2vtx = del_dlpack::get_managed_tensor_from_pyany(pyrmd2vtx)?;
    let prism2vtx = del_dlpack::get_managed_tensor_from_pyany(prism2vtx)?;
    let elem2idx_offset = del_dlpack::get_managed_tensor_from_pyany(elem2idx_offset)?;
    let idx2vtx = del_dlpack::get_managed_tensor_from_pyany(idx2vtx)?;
    let device = tet2vtx.ctx.device_type;
    //
    let num_tet = del_dlpack::get_shape_tensor(tet2vtx, 0).unwrap();
    let num_pyrmd = del_dlpack::get_shape_tensor(pyrmd2vtx, 0).unwrap();
    let num_prism = del_dlpack::get_shape_tensor(prism2vtx, 0).unwrap();
    let num_elem = del_dlpack::get_shape_tensor(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = del_dlpack::get_shape_tensor(idx2vtx, 0).unwrap();
    //
    assert_eq!(num_elem, num_tet + num_pyrmd + num_prism);
    assert_eq!(num_idx, num_tet * 4 + num_pyrmd * 5 + num_prism * 6);
    del_dlpack::check_2d_tensor::<u32>(tet2vtx, num_tet, 4, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(pyrmd2vtx, num_pyrmd, 5, device).unwrap();
    del_dlpack::check_2d_tensor::<u32>(prism2vtx, num_prism, 6, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(idx2vtx, num_idx, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let tet2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tet2vtx) }.unwrap();
            let pyrmd2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(pyrmd2vtx) }.unwrap();
            let prism2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(prism2vtx) }.unwrap();
            let elem2idx_offset =
                unsafe { del_dlpack::slice_from_tensor_mut::<u32>(elem2idx_offset) }.unwrap();
            let idx2vtx = unsafe { del_dlpack::slice_from_tensor_mut::<u32>(idx2vtx) }.unwrap();
            del_msh_cpu::mixed_mesh::to_polyhedron_mesh(
                tet2vtx,
                pyrmd2vtx,
                prism2vtx,
                elem2idx_offset,
                idx2vtx,
            );
        }
        _ => {
            todo!()
        }
    }

    Ok(())
}
