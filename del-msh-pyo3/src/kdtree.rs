use numpy::IntoPyArray;
use numpy::PyUntypedArrayMethods;
//
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    use pyo3::prelude::PyModuleMethods;
    m.add_function(wrap_pyfunction!(kdtree_build_2d, m)?)?;
    m.add_function(wrap_pyfunction!(kdtree_edge_2d, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn kdtree_build_2d<'a>(
    _py: pyo3::Python<'a>,
    vtx2xy: numpy::PyReadonlyArray2<'a, f64>) -> Bound<'a, numpy::PyArray2<usize>>
{
    type Vector = nalgebra::Vector2::<f64>;
    let num_vtx = vtx2xy.shape()[0];
    let vtx2xy = vtx2xy.as_slice().unwrap();
    let mut pairs_xy_idx = Vec::<(Vector,usize)>::new();
    for (i_vtx, xy) in vtx2xy.chunks(2).enumerate() {
        pairs_xy_idx.push( (Vector::new(xy[0], xy[1]), i_vtx) );
    }
    let mut tree = Vec::<usize>::new();
    del_msh::kdtree2::construct_kdtree(
        &mut tree, 0,
        &mut pairs_xy_idx, 0, num_vtx,
        0);
    numpy::ndarray::Array2::from_shape_vec(
        (tree.len() / 3, 3), tree).unwrap().into_pyarray_bound(_py)
}

#[pyo3::pyfunction]
fn kdtree_edge_2d<'a>(
    _py: pyo3::Python<'a>,
    tree: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xy: numpy::PyReadonlyArray2<'a, f64>,
    vmin: Vec<f64>,
    vmax: Vec<f64>)
    -> Bound<'a, numpy::PyArray3<f64>>
{
    let min = nalgebra::Vector2::<f64>::from_vec(vmin);
    let max = nalgebra::Vector2::<f64>::from_vec(vmax);
    let vtx2xy = vtx2xy.as_slice().unwrap();
    let tree = tree.as_slice().unwrap();
    let mut edge2xy = Vec::<f64>::new();
    del_msh::kdtree2::find_edges(
        &mut edge2xy,
        vtx2xy,
        tree,
        0,
        min, max,
        0);
    edge2xy.push(min.x);
    edge2xy.push(min.y);
    edge2xy.push(max.x);
    edge2xy.push(min.y);
    //
    edge2xy.push(max.x);
    edge2xy.push(min.y);
    edge2xy.push(max.x);
    edge2xy.push(max.y);
    //
    edge2xy.push(max.x);
    edge2xy.push(max.y);
    edge2xy.push(min.x);
    edge2xy.push(max.y);
    //
    edge2xy.push(min.x);
    edge2xy.push(max.y);
    edge2xy.push(min.x);
    edge2xy.push(min.y);
    numpy::ndarray::Array3::from_shape_vec(
        (edge2xy.len() / 4, 2, 2), edge2xy).unwrap().into_pyarray_bound(_py)
}