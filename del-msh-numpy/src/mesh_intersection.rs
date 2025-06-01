use numpy::ToPyArray;
//
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    // related to self-collision
    m.add_function(wrap_pyfunction!(intersection_trimesh3, m)?)?;
    m.add_function(wrap_pyfunction!(ccd_intersection_time, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(contacting_pair, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn intersection_trimesh3<'a>(
    _py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    aabbs: numpy::PyReadonlyArray2<'a, f32>,
    i_bvhnode_root: usize,
) -> (
    Bound<'a, numpy::PyArray3<f32>>,
    Bound<'a, numpy::PyArray2<usize>>,
) {
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let bvhnodes = bvhnodes.as_slice().unwrap();
    let aabbs = aabbs.as_slice().unwrap();
    type TriPair = del_msh_cpu::trimesh3_intersection::IntersectingPair<f32>;
    let pairs = if bvhnodes.is_empty() {
        del_msh_cpu::trimesh3_intersection::search_brute_force(tri2vtx, vtx2xyz)
    } else {
        let mut pairs = Vec::<TriPair>::new();
        del_msh_cpu::trimesh3_intersection::search_with_bvh_inside_branch(
            &mut pairs,
            tri2vtx,
            vtx2xyz,
            i_bvhnode_root,
            bvhnodes,
            aabbs,
        );
        pairs
    };
    let mut edge2node2xyz = Vec::<f32>::new();
    let mut edge2tri = Vec::<usize>::new();
    for pair in pairs.iter() {
        edge2node2xyz.extend(pair.p0.iter());
        edge2node2xyz.extend(pair.p1.iter());
        edge2tri.push(pair.i_tri);
        edge2tri.push(pair.j_tri);
    }
    (
        numpy::ndarray::Array3::from_shape_vec((edge2node2xyz.len() / 6, 2, 3), edge2node2xyz)
            .unwrap()
            .to_pyarray(_py),
        numpy::ndarray::Array2::from_shape_vec((edge2tri.len() / 2, 2), edge2tri)
            .unwrap()
            .to_pyarray(_py),
    )
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
fn ccd_intersection_time<'a>(
    _py: pyo3::Python<'a>,
    edge2vtx: numpy::PyReadonlyArray2<'a, usize>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f32>,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f32>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    aabbs: numpy::PyReadonlyArray2<'a, f32>,
    roots: Vec<usize>,
) -> (
    Bound<'a, numpy::PyArray2<usize>>,
    Bound<'a, numpy::PyArray1<f32>>,
) {
    let edge2vtx = edge2vtx.as_slice().unwrap();
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz0 = vtx2xyz0.as_slice().unwrap();
    let vtx2xyz1 = vtx2xyz1.as_slice().unwrap();
    let bvhnodes = bvhnodes.as_slice().unwrap();
    let aabbs = aabbs.as_slice().unwrap();
    let (intersecting_pair, intersecting_time) = if bvhnodes.is_empty() {
        del_msh_cpu::trimesh3_intersection_time::search_brute_force(
            edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1, 1.0e-8f32,
        )
    } else {
        assert_eq!(bvhnodes.len() * 2, aabbs.len());
        assert_eq!(roots.len(), 3);
        del_msh_cpu::trimesh3_intersection_time::search_with_bvh(
            edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1, bvhnodes, aabbs,
        )
    };
    (
        numpy::ndarray::Array2::from_shape_vec((intersecting_pair.len() / 3, 3), intersecting_pair)
            .unwrap()
            .to_pyarray(_py),
        numpy::ndarray::Array1::from_vec(intersecting_time).to_pyarray(_py),
    )
}

#[pyo3::pyfunction]
fn contacting_pair<'a>(
    _py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>,
    edge2vtx: numpy::PyReadonlyArray2<'a, usize>,
    threshold: f32,
) -> (
    Bound<'a, numpy::PyArray2<usize>>,
    Bound<'a, numpy::PyArray2<f32>>,
) {
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let edge2vtx = edge2vtx.as_slice().unwrap();
    let (contacting_pair, contacting_coord) =
        del_msh_cpu::trimesh3_proximity::contacting_pair(tri2vtx, vtx2xyz, edge2vtx, threshold);
    (
        numpy::ndarray::Array2::from_shape_vec((contacting_pair.len() / 3, 3), contacting_pair)
            .unwrap()
            .to_pyarray(_py),
        numpy::ndarray::Array2::from_shape_vec((contacting_coord.len() / 4, 4), contacting_coord)
            .unwrap()
            .to_pyarray(_py),
    )
}
