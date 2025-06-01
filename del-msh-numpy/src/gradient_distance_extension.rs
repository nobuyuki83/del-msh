use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
//
use pyo3::{Bound, Python};

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    // topology
    m.add_function(wrap_pyfunction!(extend_trimesh3, m)?)?;
    m.add_function(wrap_pyfunction!(extend_polyloop3, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::identity_op)]
pub fn extend_trimesh3<'a>(
    py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f64>,
    step: f64,
    niter: usize,
) -> Bound<'a, PyArray2<f64>> {
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let vtx2nrm = del_msh_cpu::trimesh3::vtx2normal(tri2vtx, vtx2xyz);
    let num_vtx = vtx2xyz.len() / 3;
    let mut a = vec![0_f64; num_vtx * 3];
    for i_vtx in 0..num_vtx {
        let mut p0 = [
            vtx2xyz[i_vtx * 3 + 0] + step * vtx2nrm[i_vtx * 3 + 0],
            vtx2xyz[i_vtx * 3 + 1] + step * vtx2nrm[i_vtx * 3 + 1],
            vtx2xyz[i_vtx * 3 + 2] + step * vtx2nrm[i_vtx * 3 + 2],
        ];
        for _ in 1..niter {
            p0 = del_msh_cpu::trimesh3::extend_avoid_intersection(tri2vtx, vtx2xyz, &p0, step);
        }
        a[i_vtx * 3 + 0] = p0[0];
        a[i_vtx * 3 + 1] = p0[1];
        a[i_vtx * 3 + 2] = p0[2];
    }
    numpy::ndarray::Array2::<f64>::from_shape_vec((num_vtx, 3), a)
        .unwrap()
        .into_pyarray(py)
}

#[pyo3::pyfunction]
pub fn extend_polyloop3<'a>(
    py: Python<'a>,
    lpvtx2xyz: PyReadonlyArray2<'a, f64>,
    step: f64,
    niter: usize,
) -> (Bound<'a, PyArray2<usize>>, Bound<'a, PyArray2<f64>>) {
    assert_eq!(lpvtx2xyz.shape()[1], 3);
    let lpvtx2bin = del_msh_cpu::polyloop3::smooth_frame(lpvtx2xyz.as_slice().unwrap());
    let (tri2vtx, vtx2xyz) = del_msh_cpu::polyloop3::tube_mesh_avoid_intersection(
        lpvtx2xyz.as_slice().unwrap(),
        &lpvtx2bin,
        step,
        niter,
    );
    let v1 = numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx)
        .unwrap()
        .into_pyarray(py);
    let v2 = numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz)
        .unwrap()
        .into_pyarray(py);
    (v1, v2)
}
