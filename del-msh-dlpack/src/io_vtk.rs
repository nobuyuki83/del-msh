use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, slice,
};
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(io_vtk_write_mix_mesh, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        io_vtk_write_points_with_velocity,
        m
    )?)?;
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
    hex2vtx: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let tet2vtx = get_tensor(tet2vtx)?;
    let pyrmd2vtx = get_tensor(pyrmd2vtx)?;
    let prism2vtx = get_tensor(prism2vtx)?;
    let hex2vtx = get_tensor(hex2vtx)?;
    //
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_tet = shape(tet2vtx, 0).unwrap();
    let num_pyrmd = shape(pyrmd2vtx, 0).unwrap();
    let num_prism = shape(prism2vtx, 0).unwrap();
    let num_hex = shape(hex2vtx, 0).unwrap();
    //
    chk2::<f32>(vtx2xyz, num_vtx, 3, dlpack::device_type_codes::CPU).unwrap();
    chk2::<u32>(tet2vtx, num_tet, 4, dlpack::device_type_codes::CPU).unwrap();
    chk2::<u32>(pyrmd2vtx, num_pyrmd, 5, dlpack::device_type_codes::CPU).unwrap();
    chk2::<u32>(prism2vtx, num_prism, 6, dlpack::device_type_codes::CPU).unwrap();
    chk2::<u32>(hex2vtx, num_hex, 8, dlpack::device_type_codes::CPU).unwrap();
    //
    let vtx2xyz = slice!(vtx2xyz, f32).unwrap();
    let tet2vtx = slice!(tet2vtx, u32).unwrap();
    let pyrmd2vtx = slice!(pyrmd2vtx, u32).unwrap();
    let prism2vtx = slice!(prism2vtx, u32).unwrap();
    let hex2vtx = slice!(hex2vtx, u32).unwrap();
    //
    let mut file = std::fs::File::create(path).expect("file not found.");
    del_msh_cpu::io_vtk::write_vtk_points(&mut file, "hoge", vtx2xyz, 3).unwrap();
    del_msh_cpu::io_vtk::write_vtk_cells_mix::<u32>(
        &mut file, tet2vtx, pyrmd2vtx, prism2vtx, hex2vtx,
    )
    .unwrap();
    Ok(())
}

#[pyfunction]
fn io_vtk_write_points_with_velocity(
    _py: Python<'_>,
    path: String,
    vtx2xyz: &Bound<'_, PyAny>,
    vtx2velocity: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let vtx2velocity = get_tensor(vtx2velocity)?;
    //
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, dlpack::device_type_codes::CPU).unwrap();
    chk2::<f32>(vtx2velocity, num_vtx, 3, dlpack::device_type_codes::CPU).unwrap();
    //
    let mut file = std::fs::File::create(path).expect("file not found.");
    del_msh_cpu::io_vtk::write_vtk_points_with_velocity(
        &mut file,
        slice!(vtx2xyz, f32).unwrap(),
        slice!(vtx2velocity, f32).unwrap(),
    )
    .unwrap();
    Ok(())
}
