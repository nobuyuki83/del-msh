use del_dlpack::pyo3;
//
use pyo3::prelude::PyModuleMethods;
use pyo3::{types::PyModule, Bound, PyResult, Python};

mod array1d;
mod edge2vtx;
mod io_cfd_mesh_txt;
mod io_nastran;
mod io_wavefront_obj;
mod mortons;
mod nbody;
mod offset_array;
mod quad_oct_tree;
mod trimesh3;
mod trimesh3_raycast;
mod vtx2elem;
mod vtx2vtx;

#[pyo3::pymodule]
#[pyo3(name = "del_msh_dlpack")]
fn del_msh_dlpack_(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(get_cuda_driver_version, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_ptx_compiler_version, m)?)?;
    trimesh3_raycast::add_functions(_py, m)?;
    trimesh3::add_functions(_py, m)?;
    edge2vtx::add_functions(_py, m)?;
    vtx2vtx::add_functions(_py, m)?;
    vtx2elem::add_functions(_py, m)?;
    mortons::add_functions(_py, m)?;
    array1d::add_functions(_py, m)?;
    quad_oct_tree::add_functions(_py, m)?;
    offset_array::add_functions(_py, m)?;
    nbody::add_functions(_py, m)?;
    io_nastran::add_functions(_py, m)?;
    io_wavefront_obj::add_functions(_py, m)?;
    io_cfd_mesh_txt::add_functions(_py, m)?;
    Ok(())
}

// --------------------------------

#[pyo3::pyfunction]
pub fn get_cuda_driver_version() -> PyResult<(u32, u32)> {
    #[cfg(feature = "cuda")]
    {
        use del_cudarc_sys::{cu, cuda_check};
        cuda_check!(cu::cuInit(0)).unwrap();
        let mut version: i32 = 0;
        cuda_check!(cu::cuDriverGetVersion(&mut version)).unwrap();
        let major = (version / 1000) as u32;
        let minor = ((version % 1000) / 10) as u32;
        return Ok((major, minor));
    }
    #[allow(unreachable_code)]
    Ok((u32::MAX, u32::MAX))
}

#[pyo3::pyfunction]
pub fn get_ptx_compiler_version() -> PyResult<(i32, i32, i32)> {
    #[cfg(feature = "cuda")]
    {
        let a = del_cudarc_sys::get_ptx_compiler_version();
        return Ok(a);
    }
    #[allow(unreachable_code)]
    Ok((-1, -1, -1))
}
