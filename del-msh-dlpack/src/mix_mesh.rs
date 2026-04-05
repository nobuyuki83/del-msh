use del_dlpack::{
    dlpack, get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape,
    check_1d_tensor as chk1, check_2d_tensor as chk2, slice, slice_mut,
};
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
    let tet2vtx = get_tensor(tet2vtx)?;
    let pyrmd2vtx = get_tensor(pyrmd2vtx)?;
    let prism2vtx = get_tensor(prism2vtx)?;
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let device = tet2vtx.ctx.device_type;
    //
    let num_tet = shape(tet2vtx, 0).unwrap();
    let num_pyrmd = shape(pyrmd2vtx, 0).unwrap();
    let num_prism = shape(prism2vtx, 0).unwrap();
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    //
    assert_eq!(num_elem, num_tet + num_pyrmd + num_prism);
    assert_eq!(num_idx, num_tet * 4 + num_pyrmd * 5 + num_prism * 6);
    chk2::<u32>(tet2vtx, num_tet, 4, device).unwrap();
    chk2::<u32>(pyrmd2vtx, num_pyrmd, 5, device).unwrap();
    chk2::<u32>(prism2vtx, num_prism, 6, device).unwrap();
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::mixed_mesh::to_polyhedron_mesh(
                slice!(tet2vtx, u32).unwrap(),
                slice!(pyrmd2vtx, u32).unwrap(),
                slice!(prism2vtx, u32).unwrap(),
                slice_mut!(elem2idx_offset, u32).unwrap(),
                slice_mut!(idx2vtx, u32).unwrap(),
            );
        }
        _ => {
            todo!()
        }
    }

    Ok(())
}
