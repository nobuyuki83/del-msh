use numpy::PyReadonlyArray1;
use pyo3::{types::PyModule, Bound, PyResult, Python};
pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(sample_uniformly_trimesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn sample_uniformly_trimesh(
    _py: Python,
    cumsum_tri2area: PyReadonlyArray1<f32>,
    val01_a: f32,
    val01_b: f32,
) -> (usize, f32, f32) {
    del_msh_cpu::trimesh::sample_uniformly(cumsum_tri2area.as_slice().unwrap(), val01_a, val01_b)
}
