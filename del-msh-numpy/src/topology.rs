use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pyfunction, types::PyModule, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(vtx2vtx_trimesh, m)?)?;
    m.add_function(wrap_pyfunction!(triangles_from_polygon_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(
        group_connected_element_uniform_polygon_mesh,
        m
    )?)?;
    Ok(())
}

#[pyfunction]
fn triangles_from_polygon_mesh<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>,
) -> Bound<'a, PyArray2<usize>> {
    let (tri2vtx, _) = del_msh_cpu::tri2vtx::from_polygon_mesh(
        elem2idx.as_slice().unwrap(),
        idx2vtx.as_slice().unwrap(),
    );
    numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx)
        .unwrap()
        .into_pyarray(py)
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

#[pyfunction]
fn group_connected_element_uniform_polygon_mesh<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
) -> (usize, Bound<'a, PyArray1<usize>>) {
    let num_node = elem2vtx.shape()[1];
    let (face2idx, idx2node) = del_msh_cpu::elem2elem::face2node_of_polygon_element(num_node);
    let elem2adjelem = del_msh_cpu::elem2elem::from_uniform_mesh(
        elem2vtx.as_slice().unwrap(),
        num_node,
        &face2idx,
        &idx2node,
        num_vtx,
    );
    let (num_group, elem2group) = del_msh_cpu::elem2group::from_uniform_mesh_with_elem2elem(
        elem2vtx.as_slice().unwrap(),
        num_node,
        &elem2adjelem,
    );
    (
        num_group,
        numpy::ndarray::Array1::from_vec(elem2group).into_pyarray(py),
    )
}
