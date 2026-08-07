use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::{types::PyModule, Bound, PyResult, Python};
pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(
        unidex_vertex_attribute_for_triangle_mesh,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn unidex_vertex_attribute_for_triangle_mesh<'a>(
    py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
) -> Bound<'a, PyArray3<f32>> {
    let tri2xyz = del_msh_cpu::unindex::unidex_vertex_attribute_for_triangle_mesh(
        tri2vtx.as_slice().unwrap().as_chunks::<3>().0,
        vtx2xyz.as_slice().unwrap().as_chunks::<3>().0,
    );
    let num_tri = tri2vtx.shape()[0];
    numpy::ndarray::Array3::from_shape_vec((num_tri, 3, 3), tri2xyz.into_flattened())
        .unwrap()
        .into_pyarray(py)
}
