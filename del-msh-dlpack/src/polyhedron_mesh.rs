#[cfg(feature = "cuda")]
use del_cudarc_sys::{cu::CUdeviceptr, CuVec, LaunchConfig};

use del_dlpack::dlpack;
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(polyhedron_mesh_elem2volume, m)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn polyhedron_mesh_elem2volume(
    _py: Python<'_>,
    elem2idx_offset: &Bound<'_, PyAny>,
    idx2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    elem2volume: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let elem2idx_offset = del_dlpack::get_managed_tensor_from_pyany(elem2idx_offset)?;
    let idx2vtx = del_dlpack::get_managed_tensor_from_pyany(idx2vtx)?;
    let vtx2xyz = del_dlpack::get_managed_tensor_from_pyany(vtx2xyz)?;
    let elem2volume = del_dlpack::get_managed_tensor_from_pyany(elem2volume)?;
    //
    let device = elem2idx_offset.ctx.device_type;
    let num_elem = del_dlpack::get_shape_tensor(elem2idx_offset, 0).unwrap() - 1;
    let num_idx = del_dlpack::get_shape_tensor(idx2vtx, 0).unwrap();
    let num_vtx = del_dlpack::get_shape_tensor(vtx2xyz, 0).unwrap();
    //
    del_dlpack::check_1d_tensor::<u32>(elem2idx_offset, num_elem + 1, device).unwrap();
    del_dlpack::check_1d_tensor::<u32>(idx2vtx, num_idx, device).unwrap();
    del_dlpack::check_2d_tensor::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    del_dlpack::check_1d_tensor::<f32>(elem2volume, num_elem, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let elem2idx_offset =
                unsafe { del_dlpack::slice_from_tensor::<u32>(elem2idx_offset) }.unwrap();
            let idx2vtx = unsafe { del_dlpack::slice_from_tensor::<u32>(idx2vtx) }.unwrap();
            let vtx2xyz = unsafe { del_dlpack::slice_from_tensor::<f32>(vtx2xyz) }.unwrap();
            let elem2volume =
                unsafe { del_dlpack::slice_from_tensor_mut::<f32>(elem2volume) }.unwrap();
            del_msh_cpu::polyhedron_mesh::elem2volume(
                elem2idx_offset,
                idx2vtx,
                vtx2xyz,
                1,
                elem2volume,
            );
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
