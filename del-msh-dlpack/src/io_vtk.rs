use del_dlpack::dlpack;
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(io_vtk_write_mix_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn io_vtk_write_mix_mesh(
    _py: Python<'_>,
    path: String,
    vtx2xyz: &Bound<'_, PyAny>,
    tet2vtx: &Bound<'_, PyAny>,
    pyrmd2vtx: &Bound<'_, PyAny>,
    prism2vtx: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let tet2vtx = del_dlpack::get_managed_tensor_from_pyany(tet2vtx)?;
    let pyrmd2vtx = del_dlpack::get_managed_tensor_from_pyany(pyrmd2vtx)?;
    let prism2vtx = del_dlpack::get_managed_tensor_from_pyany(prism2vtx)?;
    //
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    let num_tet = del_dlpack::get_shape_tensor(tet2vtx, 0).unwrap();
    let num_pyrmd = del_dlpack::get_shape_tensor(pyrmd2vtx, 0).unwrap();
    let num_prism = del_dlpack::get_shape_tensor(prism2vtx, 0).unwrap();
    //
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, dlpack::device_type_codes::CPU)
        .unwrap();
    del_dlpack::check_2d_tensor::<u32>(tet2vtx, num_tet, 4, dlpack::device_type_codes::CPU)
        .unwrap();
    del_dlpack::check_2d_tensor::<u32>(pyrmd2vtx, num_pyrmd, 5, dlpack::device_type_codes::CPU)
        .unwrap();
    del_dlpack::check_2d_tensor::<u32>(prism2vtx, num_prism, 6, dlpack::device_type_codes::CPU)
        .unwrap();
    //
    let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz) }.unwrap();
    let tet2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(tet2vtx) }.unwrap();
    let pyrmd2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(pyrmd2vtx) }.unwrap();
    let prism2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(prism2vtx) }.unwrap();
    //
    let mut file = std::fs::File::create(path).expect("file not found.");
    del_msh_cpu::io_vtk::write_vtk_points(&mut file, "hoge", vtx2xyz, 3).unwrap();
    del_msh_cpu::io_vtk::write_vtk_cells_mix::<u32>(&mut file, tet2vtx, pyrmd2vtx, prism2vtx)
        .unwrap();
    Ok(())
}
