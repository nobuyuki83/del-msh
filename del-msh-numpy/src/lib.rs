use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{types::PyModule, Bound, PyResult, Python};

mod bvh;
mod dijkstra;
mod dtri;
mod edge2vtx;
mod elem2elem;
mod extract;
mod gradient_distance_extension;
mod io;
mod kdtree;
mod mesh_intersection;
mod polyloop;
mod primitive;
mod sampling;
mod topology;
mod trimesh3_search;
mod unify_index;
mod unindex;
mod vtx2area;

/* ------------------------ */

#[pyo3::pymodule]
#[pyo3(name = "del_msh_numpy")]
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
            2 => del_msh_cpu::trimesh2::tri2area(
                tri2vtx.as_slice().unwrap(),
                vtx2xyz.as_slice().unwrap(),
            ),
            3 => del_msh_cpu::trimesh3::tri2area(
                tri2vtx.as_slice().unwrap(),
                vtx2xyz.as_slice().unwrap(),
            ),
            _ => {
                panic!();
            }
        };
        numpy::ndarray::Array1::from_shape_vec(tri2vtx.shape()[0], tri2area)
            .unwrap()
            .into_pyarray(py)
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
            2 => del_msh_cpu::trimesh2::tri2circumcenter(
                tri2vtx.as_slice().unwrap(),
                vtx2xyz.as_slice().unwrap(),
            ),
            _ => {
                panic!();
            }
        };
        numpy::ndarray::Array2::from_shape_vec((tri2vtx.shape()[0], num_dim), tri2cc)
            .unwrap()
            .into_pyarray(py)
    }

    Ok(())
}
