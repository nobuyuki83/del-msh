use numpy::{IntoPyArray, PyArray2};
use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(wrap_pyfunction!(torus_meshtri3, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_meshtri3, m)?)?;
    m.add_function(wrap_pyfunction!(cylinder_closed_end_meshtri3, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_meshtri3, m)?)?;
    m.add_function(wrap_pyfunction!(trimesh3_hemisphere_zup, m)?)?;
    Ok(())
}

#[pyfunction]
fn torus_meshtri3(
    py: Python,
    radius: f64,
    radius_tube: f64,
    nlg: usize,
    nlt: usize,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f64>>) {
    let (tri2vtx, vtx2xyz) =
        del_msh_cpu::trimesh3_primitive::torus_zup::<usize, f64>(radius, radius_tube, nlg, nlt);
    let v = numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz).unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx).unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn capsule_meshtri3(
    py: Python,
    r: f64,
    l: f64,
    nc: usize,
    nr: usize,
    nl: usize,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f64>>) {
    let (tri_vtx, vtx_xyz) = del_msh_cpu::trimesh3_primitive::capsule_yup::<f64>(r, l, nc, nr, nl);
    let v = numpy::ndarray::Array2::from_shape_vec(
        (vtx_xyz.len() / 3, 3),
        Vec::from(vtx_xyz.as_slice()),
    )
    .unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec(
        (tri_vtx.len() / 3, 3),
        Vec::from(tri_vtx.as_slice()),
    )
    .unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn cylinder_closed_end_meshtri3(
    py: Python,
    radius: f64,
    length: f64,
    ndiv_circumference: usize,
    ndiv_length: usize,
    is_closed_end: bool,
    is_center: bool,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f64>>) {
    let (tri2vtx, vtx2xyz) = if is_closed_end {
        del_msh_cpu::trimesh3_primitive::cylinder_closed_end_yup::<f64>(
            radius,
            length,
            ndiv_circumference,
            ndiv_length,
            is_center,
        )
    } else {
        del_msh_cpu::trimesh3_primitive::cylinder_open_end_yup::<f64>(
            ndiv_circumference,
            ndiv_length,
            radius,
            length,
            is_center,
        )
    };
    let v = numpy::ndarray::Array2::from_shape_vec(
        (vtx2xyz.len() / 3, 3),
        Vec::from(vtx2xyz.as_slice()),
    )
    .unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec(
        (tri2vtx.len() / 3, 3),
        Vec::from(tri2vtx.as_slice()),
    )
    .unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn sphere_meshtri3(
    py: Python,
    r: f32,
    nr: usize,
    nl: usize,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f32>>) {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::sphere_yup(r, nr, nl);
    let v = numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz).unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx).unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn trimesh3_hemisphere_zup(
    py: Python,
    r: f32,
    nr: usize,
    nl: usize,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f32>>) {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::hemisphere_zup(r, nr, nl);
    (
        numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz)
            .unwrap()
            .into_pyarray(py),
    )
}
