use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{pyfunction, Bound, Python};

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(polyline_vtx2xyz_from_helix, m)?)?;
    m.add_function(wrap_pyfunction!(polyline_vtx2framex_from_vtx2xyz, m)?)?;
    m.add_function(wrap_pyfunction!(polyline_vtx2vtx_rods, m)?)?;
    Ok(())
}

#[pyfunction]
fn polyline_vtx2xyz_from_helix(
    py: Python,
    num_vtx: usize,
    elen: f32,
    rad: f32,
    pitch: f32,
) -> Bound<PyArray2<f32>> {
    let vtx2xyz = del_msh_cpu::polyline3::helix(num_vtx, elen, rad, pitch);
    let vtx2xyz = numpy::ndarray::Array2::from_shape_vec(
        (vtx2xyz.len() / 3, 3),
        Vec::from(vtx2xyz.as_slice()),
    )
    .unwrap();
    use numpy::IntoPyArray;
    vtx2xyz.into_pyarray(py)
}

#[pyfunction]
fn polyline_vtx2framex_from_vtx2xyz<'a>(
    py: Python<'a>,
    vtx2xyz: numpy::PyReadonlyArray2<f32>,
) -> Bound<'a, PyArray2<f32>> {
    let vtx2framex = del_msh_cpu::polyline3::vtx2framex(vtx2xyz.as_slice().unwrap());
    let vtx2framex = numpy::ndarray::Array2::from_shape_vec(
        (vtx2framex.len() / 3, 3),
        Vec::from(vtx2framex.as_slice()),
    )
    .unwrap();
    use numpy::IntoPyArray;
    vtx2framex.into_pyarray(py)
}

#[pyfunction]
fn polyline_vtx2vtx_rods<'a>(
    py: Python<'a>,
    hair2root: numpy::PyReadonlyArray1<usize>,
) -> (Bound<'a, PyArray1<usize>>, Bound<'a, PyArray1<usize>>) {
    let (vtx2idx, idx2vtx) = del_msh_cpu::polyline::vtx2vtx_rods(hair2root.as_slice().unwrap());
    (vtx2idx.into_pyarray(py), idx2vtx.into_pyarray(py))
}
