use del_dlpack::make_capsule_from_vec as capsule;
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(io_off_load_tri_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn io_off_load_tri_mesh(
    py: Python<'_>,
    path: String,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_off::load_as_tri_mesh::<_, u32, f32>(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let num_tri = tri2vtx.len() as i64;
    let num_vtx = vtx2xyz.len() as i64;
    let tri2vtx = capsule(py, vec![num_tri, 3], tri2vtx.into_flattened());
    let vtx2xyz = capsule(py, vec![num_vtx, 3], vtx2xyz.into_flattened());
    Ok((tri2vtx, vtx2xyz))
}
