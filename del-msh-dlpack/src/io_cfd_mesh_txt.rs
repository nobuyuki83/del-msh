use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(io_cfd_mesh_txt_load, m)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn io_cfd_mesh_txt_load(
    py: Python<'_>,
    path: String,
) -> PyResult<(
    pyo3::Py<PyAny>,
    pyo3::Py<PyAny>,
    pyo3::Py<PyAny>,
    pyo3::Py<PyAny>,
)> {
    let data = del_msh_cpu::io_cfd_mesh_txt::read::<_, u32>(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let vtx2xyz_cap = del_dlpack::make_capsule_from_vec(
        py,
        vec![(data.vtx2xyz.len() as i64) / 3, 3],
        data.vtx2xyz,
    );
    let tet2vtx_cap = del_dlpack::make_capsule_from_vec(
        py,
        vec![(data.tet2vtx.len() as i64) / 4, 4],
        data.tet2vtx,
    );
    let pyrmd2vtx_cap = del_dlpack::make_capsule_from_vec(
        py,
        vec![(data.pyrmd2vtx.len() as i64) / 5, 5],
        data.pyrmd2vtx,
    );
    let prism2vtx_cap = del_dlpack::make_capsule_from_vec(
        py,
        vec![(data.prism2vtx.len() as i64) / 6, 6],
        data.prism2vtx,
    );
    Ok((vtx2xyz_cap, tet2vtx_cap, pyrmd2vtx_cap, prism2vtx_cap))
}
