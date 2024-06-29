use numpy::{
    IntoPyArray,
    PyReadonlyArray2,
    PyArray1, PyArray2};
use numpy::PyUntypedArrayMethods;
use pyo3::{types::PyModule, PyResult, Python, Bound};

mod topology;
mod primitive;
mod io;
mod unify_index;
mod unindex;
mod dijkstra;
mod sampling;
mod extract;
mod trimesh3_search;
mod edge2vtx;
mod elem2elem;
mod dtri;
mod polyloop;
mod bvh;
mod kdtree;
mod mesh_intersection;
mod gradient_distance_extension;
mod vtx2area;

/* ------------------------ */

#[pyo3::pymodule]
#[pyo3(name = "del_msh")]
fn del_msh_(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    topology::add_functions(_py, m)?;
    edge2vtx::add_functions(_py, m)?;
    elem2elem::add_functions(_py, m)?;
    unify_index::add_functions(_py, m)?;
    unindex::add_functions(_py, m)?;
    dijkstra::add_functions(_py, m)?;
    primitive::add_functions(_py, m)?;
    io::add_functions(_py, m)?;
    sampling::add_functions(_py, m)?;
    extract::add_functions(_py, m)?;
    trimesh3_search::add_functions(_py, m)?;
    dtri::add_functions(_py, m)?;
    polyloop::add_functions(_py, m)?;
    bvh::add_functions(_py, m)?;
    kdtree::add_functions(_py, m)?;
    mesh_intersection::add_functions(_py, m)?;
    gradient_distance_extension::add_functions(_py, m)?;
    vtx2area::add_functions(_py, m)?;

    #[pyfn(m)]
    pub fn areas_of_triangles_of_mesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f32>,
    ) -> Bound<'a, PyArray1<f32>> {
        assert!(tri2vtx.is_c_contiguous());
        assert!(vtx2xyz.is_c_contiguous());
        let tri2area = match vtx2xyz.shape()[1] {
            2 => {
                del_msh::trimesh2::tri2area(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            3 => {
                del_msh::trimesh3::tri2area(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            _ => { panic!(); }
        };
        numpy::ndarray::Array1::from_shape_vec(
            tri2vtx.shape()[0], tri2area).unwrap().into_pyarray_bound(py)
    }

    #[pyfn(m)]
    pub fn circumcenters_of_triangles_of_mesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f32>,
    ) -> pyo3::Bound<'a, PyArray2<f32>> {
        assert!(tri2vtx.is_c_contiguous());
        assert!(vtx2xyz.is_c_contiguous());
        let num_dim = vtx2xyz.shape()[1];
        let tri2cc = match num_dim {
            2 => {
                del_msh::trimesh2::tri2circumcenter(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            _ => { panic!(); }
        };
        numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.shape()[0],num_dim), tri2cc).unwrap().into_pyarray_bound(py)
    }

    Ok(())
}