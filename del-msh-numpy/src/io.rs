use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, Bound, PyObject, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(wrap_pyfunction!(load_wavefront_obj, m)?)?;
    m.add_function(wrap_pyfunction!(load_wavefront_obj_as_triangle_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(load_nastran_as_triangle_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(load_off_as_triangle_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(save_wavefront_obj_for_uniform_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn load_wavefront_obj(
    py: Python,
    path_file: String,
) -> (
    Bound<PyArray2<f32>>,
    Bound<PyArray2<f32>>,
    Bound<PyArray2<f32>>,
    Bound<PyArray1<usize>>,
    Bound<PyArray1<usize>>,
    Bound<PyArray1<usize>>,
    Bound<PyArray1<usize>>,
    Bound<PyArray1<usize>>,
    PyObject,
    Bound<PyArray1<usize>>,
    PyObject,
    PyObject,
) {
    let mut obj = del_msh_cpu::io_obj::WavefrontObj::<usize, f32>::new();
    if let Err(str) = obj.load(&path_file) {
        dbg!(str);
        panic!();
    }
    use pyo3::IntoPyObject;
    (
        numpy::ndarray::Array2::from_shape_vec((obj.vtx2xyz.len() / 3, 3), obj.vtx2xyz)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((obj.vtx2uv.len() / 2, 2), obj.vtx2uv)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((obj.vtx2nrm.len() / 3, 3), obj.vtx2nrm)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.elem2idx).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.idx2vtx_xyz).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.idx2vtx_uv).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.idx2vtx_nrm).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.elem2group).into_pyarray(py),
        obj.group2name.into_pyobject(py).unwrap().into(),
        numpy::ndarray::Array1::from_vec(obj.elem2mtl).into_pyarray(py),
        obj.mtl2name.into_pyobject(py).unwrap().into(),
        obj.mtl_file_name.into_pyobject(py).unwrap().into(),
    )
}

#[pyfunction]
pub fn load_wavefront_obj_as_triangle_mesh(
    py: Python,
    path_file: String,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f32>>) {
    let Ok((tri2vtx, vtx2xyz)) = del_msh_cpu::io_obj::load_tri_mesh(path_file, None) else {
        todo!()
    };
    (
        numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz)
            .unwrap()
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn load_nastran_as_triangle_mesh(
    py: Python,
    path_file: String,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f32>>) {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_nas::load_tri_mesh(path_file);
    (
        numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz)
            .unwrap()
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn load_off_as_triangle_mesh(
    py: Python,
    path_file: String,
) -> (Bound<PyArray2<usize>>, Bound<PyArray2<f32>>) {
    let Ok((tri2vtx, vtx2xyz)) = del_msh_cpu::io_off::load_as_tri_mesh(path_file) else {
        todo!()
    };
    (
        numpy::ndarray::Array2::from_shape_vec((tri2vtx.len() / 3, 3), tri2vtx)
            .unwrap()
            .into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((vtx2xyz.len() / 3, 3), vtx2xyz)
            .unwrap()
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn save_wavefront_obj_for_uniform_mesh<'a>(
    _py: Python<'a>,
    path_file: String,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
) {
    let num_dim = vtx2xyz.shape()[1];
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let _ = del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(path_file, tri2vtx, vtx2xyz, num_dim);
}
