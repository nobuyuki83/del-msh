use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_primitive_torus_zup, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_primitive_sphere_yup, m)?)?;
    Ok(())
}

#[pyfunction]
fn trimesh3_primitive_torus_zup(
    py: Python<'_>,
    major_radius: f32,
    minor_radius: f32,
    ndiv_major: usize,
    ndiv_minor: usize,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::torus_zup::<u32, f32>(
        major_radius,
        minor_radius,
        ndiv_major,
        ndiv_minor,
    );
    let tri2vtx_cap =
        del_dlpack::make_capsule_from_vec(py, vec![tri2vtx.len() as i64 / 3, 3], tri2vtx);
    let vtx2xyz_cap =
        del_dlpack::make_capsule_from_vec(py, vec![vtx2xyz.len() as i64 / 3, 3], vtx2xyz);
    Ok((tri2vtx_cap, vtx2xyz_cap))
}

#[pyfunction]
fn trimesh3_primitive_sphere_yup(
    py: Python<'_>,
    radius: f32,
    ndiv_longtitude: usize,
    ndiv_latitude: usize,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(
        radius,
        ndiv_longtitude,
        ndiv_latitude,
    );
    let tri2vtx_cap =
        del_dlpack::make_capsule_from_vec(py, vec![tri2vtx.len() as i64 / 3, 3], tri2vtx);
    let vtx2xyz_cap =
        del_dlpack::make_capsule_from_vec(py, vec![vtx2xyz.len() as i64 / 3, 3], vtx2xyz);
    Ok((tri2vtx_cap, vtx2xyz_cap))
}
