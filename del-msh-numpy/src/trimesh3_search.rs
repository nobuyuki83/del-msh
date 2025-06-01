use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{types::PyModule, Bound, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(first_intersection_ray_meshtri3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pick_vertex_meshtri3, m)?)?;
    Ok(())
}

fn squared_dist(p0: &[f32], p1: &[f32]) -> f32 {
    (p0[0] - p1[0]) * (p0[0] - p1[0])
        + (p0[1] - p1[1]) * (p0[1] - p1[1])
        + (p0[2] - p1[2]) * (p0[2] - p1[2])
}

#[pyo3::pyfunction]
fn first_intersection_ray_meshtri3<'a>(
    py: Python<'a>,
    src: PyReadonlyArray1<'a, f32>,
    dir: PyReadonlyArray1<'a, f32>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
) -> (Bound<'a, PyArray1<f32>>, i64) {
    let ray_org = src.as_slice().unwrap().try_into().unwrap();
    let ray_dir = dir.as_slice().unwrap().try_into().unwrap();
    let res = del_msh_cpu::trimesh3_search_bruteforce::first_intersection_ray(
        ray_org,
        ray_dir,
        tri2vtx.as_slice().unwrap(),
        vtx2xyz.as_slice().unwrap(),
    );
    match res {
        None => {
            let a = PyArray1::<f32>::zeros(py, 3, true);
            (a, -1)
        }
        Some(postri) => {
            let t = postri.0;
            let p = [
                t * ray_dir[0] + ray_org[0],
                t * ray_dir[1] + ray_org[1],
                t * ray_dir[2] + ray_org[2],
            ];
            let a = PyArray1::<f32>::from_slice(py, &p);
            (a, postri.1 as i64)
        }
    }
}

#[pyo3::pyfunction]
fn pick_vertex_meshtri3<'a>(
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
    src: PyReadonlyArray1<'a, f32>,
    dir: PyReadonlyArray1<'a, f32>,
) -> i64 {
    let ray_org = src.as_slice().unwrap().try_into().unwrap();
    let ray_dir = dir.as_slice().unwrap().try_into().unwrap();
    let res = del_msh_cpu::trimesh3_search_bruteforce::first_intersection_ray(
        ray_org,
        ray_dir,
        tri2vtx.as_slice().unwrap(),
        vtx2xyz.as_slice().unwrap(),
    );
    match res {
        None => -1,
        Some(postri) => {
            let t = postri.0;
            let idx_tri = postri.1;
            let i0 = tri2vtx.get([idx_tri, 0]).unwrap();
            let i1 = tri2vtx.get([idx_tri, 1]).unwrap();
            let i2 = tri2vtx.get([idx_tri, 2]).unwrap();
            let q0 = &vtx2xyz.as_slice().unwrap()[i0 * 3..i0 * 3 + 3];
            let q1 = &vtx2xyz.as_slice().unwrap()[i1 * 3..i1 * 3 + 3];
            let q2 = &vtx2xyz.as_slice().unwrap()[i2 * 3..i2 * 3 + 3];
            let pos = [
                t * ray_dir[0] + ray_org[0],
                t * ray_dir[1] + ray_org[1],
                t * ray_dir[2] + ray_org[2],
            ];
            let d0 = squared_dist(&pos, q0);
            let d1 = squared_dist(&pos, q1);
            let d2 = squared_dist(&pos, q2);
            if d0 <= d1 && d0 <= d2 {
                return *i0 as i64;
            }
            if d1 <= d2 && d1 <= d0 {
                return *i1 as i64;
            }
            if d2 <= d0 && d2 <= d1 {
                return *i2 as i64;
            }
            -1
        }
    }
}
