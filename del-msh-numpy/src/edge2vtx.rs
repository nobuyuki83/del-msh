//! functions to find elem-to-vertex relationship (connectivity) for general to mesh types

use numpy::IntoPyArray;
//
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(edge2vtx_uniform_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(edge2vtx_polygon_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(boundaryedge2vtx_triangle_mesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn edge2vtx_uniform_mesh<'a>(
    py: pyo3::Python<'a>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
) -> Bound<'a, numpy::PyArray2<usize>> {
    // TODO: make this function general to uniform mesh (currently only triangle mesh)
    let mshline = del_msh_cpu::edge2vtx::from_uniform_mesh_with_specific_edges(
        elem2vtx.as_slice().unwrap(),
        3,
        &[0, 1, 1, 2, 2, 0],
        num_vtx,
    );
    numpy::ndarray::Array2::from_shape_vec((mshline.len() / 2, 2), mshline)
        .unwrap()
        .into_pyarray(py)
}

#[pyo3::pyfunction]
fn edge2vtx_polygon_mesh<'a>(
    py: pyo3::Python<'a>,
    elem2idx: numpy::PyReadonlyArray1<'a, usize>,
    idx2vtx: numpy::PyReadonlyArray1<'a, usize>,
    num_vtx: usize,
) -> Bound<'a, numpy::PyArray2<usize>> {
    let mshline = del_msh_cpu::edge2vtx::from_polygon_mesh(
        elem2idx.as_slice().unwrap(),
        idx2vtx.as_slice().unwrap(),
        num_vtx,
    );
    numpy::ndarray::Array2::from_shape_vec((mshline.len() / 2, 2), mshline)
        .unwrap()
        .into_pyarray(py)
}

#[pyo3::pyfunction]
fn boundaryedge2vtx_triangle_mesh<'a>(
    py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
) -> (
    Bound<'a, numpy::PyArray2<usize>>,
    Bound<'a, numpy::PyArray2<usize>>,
) {
    // let num_node = tri2vtx.shape()[1];
    // let num_tri = tri2vtx.shape()[0];
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let (bedge2vtx, tri2tri) = del_msh_cpu::trimesh_topology::boundaryedge2vtx(tri2vtx, num_vtx);
    (
        numpy::ndarray::Array2::from_shape_vec((bedge2vtx.len() / 2, 2), bedge2vtx)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((tri2tri.len() / 3, 3), tri2tri)
            .unwrap()
            .into_pyarray(py),
    )
}
