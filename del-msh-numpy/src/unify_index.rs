use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::{types::PyModule, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(unify_two_indices_of_triangle_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(unify_two_indices_of_polygon_mesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::type_complexity)]
fn unify_two_indices_of_triangle_mesh<'a>(
    py: Python<'a>,
    tri2vtxa: PyReadonlyArrayDyn<'a, usize>,
    tri2vtxb: PyReadonlyArrayDyn<'a, usize>,
) -> (
    Bound<'a, PyArray2<usize>>,
    Bound<'a, PyArray1<usize>>,
    Bound<'a, PyArray1<usize>>,
) {
    let (tri2uni, uni2vtxxyz, uni2vtxuv) =
        del_msh_cpu::unify_index::unify_two_indices_of_triangle_mesh(
            tri2vtxa.as_slice().unwrap(),
            tri2vtxb.as_slice().unwrap(),
        );
    (
        numpy::ndarray::Array2::from_shape_vec((tri2uni.len() / 3, 3), tri2uni)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxxyz).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxuv).into_pyarray(py),
    )
}

#[pyo3::pyfunction]
#[allow(clippy::type_complexity)]
fn unify_two_indices_of_polygon_mesh<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArrayDyn<'a, usize>,
    idx2vtxa: PyReadonlyArrayDyn<'a, usize>,
    idx2vtxb: PyReadonlyArrayDyn<'a, usize>,
) -> (
    Bound<'a, PyArray1<usize>>,
    Bound<'a, PyArray1<usize>>,
    Bound<'a, PyArray1<usize>>,
) {
    let (idx2uni, uni2vtxa, uni2vtxb) = del_msh_cpu::unify_index::unify_two_indices_of_polygon_mesh(
        elem2idx.as_slice().unwrap(),
        idx2vtxa.as_slice().unwrap(),
        idx2vtxb.as_slice().unwrap(),
    );
    (
        numpy::ndarray::Array1::from_vec(idx2uni).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxa).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxb).into_pyarray(py),
    )
}
