use del_dlpack::dlpack;
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(io_wavefront_obj_save_tri_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn io_wavefront_obj_save_tri_mesh(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<()> {
    let tri2vtx = del_dlpack::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let num_tri = del_dlpack::get_shape_tensor(tri2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<u32>(tri2vtx, num_tri, 3, dlpack::device_type_codes::CPU)
        .unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, dlpack::device_type_codes::CPU)
        .unwrap();
    //
    let tri2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tri2vtx) }.unwrap();
    let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz) }.unwrap();
    del_msh_cpu::io_wavefront_obj::save_tri2vtx_vtx2xyz(path, tri2vtx, vtx2xyz, 3).unwrap();
    Ok(())
}
