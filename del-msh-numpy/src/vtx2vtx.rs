use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{pyfunction, types::PyModule, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(vtx2vtx_trimesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn vtx2vtx_trimesh<'a>(
    py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
    is_self: bool,
) -> (Bound<'a, PyArray1<usize>>, Bound<'a, PyArray1<usize>>) {
    assert_eq!(tri2vtx.shape()[1], 3);
    let (vtx2idx, idx2vtx) =
        del_msh_cpu::vtx2vtx::from_uniform_mesh(tri2vtx.as_slice().unwrap(), 3, num_vtx, is_self);
    (
        numpy::ndarray::Array1::from_vec(vtx2idx).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(idx2vtx).into_pyarray(py),
    )
}
