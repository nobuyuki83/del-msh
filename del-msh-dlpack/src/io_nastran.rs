use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(io_nastran_load_tri_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn io_nastran_load_tri_mesh(
    py: Python<'_>,
    path: String,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_nastran::load_tri_mesh::<_, u32>(path);
    let tri2vtx_cap =
        del_dlpack::make_capsule_from_vec(py, vec![(tri2vtx.len() as i64) / 3, 3], tri2vtx);
    let vtx2xyz_cap =
        del_dlpack::make_capsule_from_vec(py, vec![(vtx2xyz.len() as i64) / 3, 3], vtx2xyz);
    Ok((tri2vtx_cap, vtx2xyz_cap))
}
