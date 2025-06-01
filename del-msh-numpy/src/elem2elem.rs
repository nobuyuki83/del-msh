//! functions to find elem-to-elem relationship (element adjacency) for general to mesh types

use numpy::IntoPyArray;
use numpy::PyUntypedArrayMethods;
//
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(
        elem2elem_uniform_mesh_polygon_indexing,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        elem2elem_uniform_mesh_simplex_indexing,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn elem2elem_uniform_mesh_polygon_indexing<'a>(
    py: pyo3::Python<'a>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
) -> Bound<'a, numpy::PyArray2<usize>> {
    assert!(elem2vtx.is_c_contiguous());
    let num_node = elem2vtx.shape()[1];
    let (face2idx, idx2node) = del_msh_cpu::elem2elem::face2node_of_polygon_element(num_node);
    let elem2elem = del_msh_cpu::elem2elem::from_uniform_mesh(
        elem2vtx.as_slice().unwrap(),
        num_node,
        &face2idx,
        &idx2node,
        num_vtx,
    );
    assert_eq!(elem2vtx.len(), elem2elem.len());
    numpy::ndarray::Array2::from_shape_vec((elem2elem.len() / num_node, num_node), elem2elem)
        .unwrap()
        .into_pyarray(py)
}

#[pyo3::pyfunction]
fn elem2elem_uniform_mesh_simplex_indexing<'a>(
    py: pyo3::Python<'a>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
) -> Bound<'a, numpy::PyArray2<usize>> {
    assert!(elem2vtx.is_c_contiguous());
    let num_node = elem2vtx.shape()[1];
    let (face2idx, idx2node) = del_msh_cpu::elem2elem::face2node_of_simplex_element(num_node);
    let elsuel = del_msh_cpu::elem2elem::from_uniform_mesh(
        elem2vtx.as_slice().unwrap(),
        num_node,
        &face2idx,
        &idx2node,
        num_vtx,
    );
    assert_eq!(elem2vtx.len(), elsuel.len());
    numpy::ndarray::Array2::from_shape_vec((elsuel.len() / num_node, num_node), elsuel)
        .unwrap()
        .into_pyarray(py)
}
