//! area (volume) per vertex for mesh with various element types

use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
//
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(vtx2area_from_uniformmesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn vtx2area_from_uniformmesh<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
) -> Bound<'a, PyArray1<f32>> {
    assert!(elem2vtx.is_c_contiguous());
    assert!(vtx2xyz.is_c_contiguous());
    let num_dim = vtx2xyz.shape()[1];
    let num_node = elem2vtx.shape()[1];
    let elem2vtx = elem2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let vtx2area = if num_node == 3 {
        // triangle mesh
        if num_dim == 2 {
            del_msh_cpu::trimesh2::vtx2area(elem2vtx, vtx2xyz)
        } else if num_dim == 3 {
            del_msh_cpu::trimesh3::vtx2area(elem2vtx, vtx2xyz)
        } else {
            panic!();
        }
    } else {
        panic!();
    };
    numpy::ndarray::Array1::from_vec(vtx2area).into_pyarray(py)
}
