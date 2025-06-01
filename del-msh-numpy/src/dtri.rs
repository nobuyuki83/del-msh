use numpy::PyArray2;
use numpy::PyUntypedArrayMethods;
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(tesselation2d, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn tesselation2d<'a>(
    py: pyo3::Python<'a>,
    vtx2xy_in: numpy::PyReadonlyArray2<'a, f32>,
    resolution_edge: f32,
    resolution_face: f32,
) -> (Bound<'a, PyArray2<usize>>, Bound<'a, PyArray2<f32>>) {
    let num_vtx = vtx2xy_in.shape()[0];
    let vtx2xy_in = vtx2xy_in.as_slice().unwrap();
    let mut loop2idx = vec![0, num_vtx];
    let mut idx2vtx = Vec::<usize>::from_iter(0..num_vtx);
    let mut vtx2xy = vtx2xy_in.chunks(2).map(|v| [v[0], v[1]]).collect();
    //
    if resolution_edge > 0. {
        // resample edge edge
        del_msh_cpu::polyloop::resample_multiple_loops_remain_original_vtxs(
            &mut loop2idx,
            &mut idx2vtx,
            &mut vtx2xy,
            resolution_edge,
        );
    }
    //
    let (mut tri2pnt, mut tri2tri, mut pnt2tri) =
        del_msh_cpu::trimesh2_dynamic::triangulate_single_connected_shape(
            &mut vtx2xy,
            &loop2idx,
            &idx2vtx,
        );
    // ----------------------------------------
    if resolution_face > 1.0e-10 {
        let nvtx = vtx2xy.len();
        let mut vtx2flag = vec![0; nvtx];
        let mut tri2flag = vec![0; tri2pnt.len() / 3];
        del_msh_cpu::trimesh2_dynamic::add_points_uniformly(
            del_msh_cpu::trimesh2_dynamic::MeshForTopologicalChange {
                tri2vtx: &mut tri2pnt,
                tri2tri: &mut tri2tri,
                vtx2tri: &mut pnt2tri,
                vtx2xy: &mut vtx2xy,
            },
            &mut vtx2flag,
            &mut tri2flag,
            nvtx,
            0,
            resolution_face,
        );
    }
    // ----------------------------------------
    let mut vtx2xy_out = Vec::<f32>::new();
    for xy in vtx2xy.iter() {
        vtx2xy_out.push(xy[0]);
        vtx2xy_out.push(xy[1]);
    }
    use numpy::IntoPyArray;
    (
        numpy::ndarray::Array2::from_shape_vec((tri2pnt.len() / 3, 3), tri2pnt)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((vtx2xy_out.len() / 2, 2), vtx2xy_out)
            .unwrap()
            .into_pyarray(py),
    )
}
