use del_dlpack::{
    dlpack, get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape,
    check_1d_tensor as chk1, check_2d_tensor as chk2, slice, slice_mut,
};
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(polyhedron_mesh_elem2volume, m)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_elem2volume(
    _py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    elem2volume: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let elem2idx_offset = get_tensor(elem2idx_offset)?;
    let idx2vtx = get_tensor(idx2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let elem2volume = get_tensor(elem2volume)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = shape(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = shape(idx2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    //
    chk1::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    chk1::<u32>(idx2vtx, num_idx, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk1::<f32>(elem2volume, num_elem, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::polyhedron_mesh::elem2volume(
                slice!(elem2idx_offset, u32).unwrap(),
                slice!(idx2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                1,
                slice_mut!(elem2volume, f32).unwrap(),
            );
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
