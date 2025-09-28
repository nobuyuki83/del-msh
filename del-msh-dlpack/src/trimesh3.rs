use pyo3::prelude::PyModule;
use pyo3::{Bound, PyAny, PyResult, Python};
use std::slice::from_raw_parts;

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_tri2normal, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(trimesh3_bwd_tri2normal, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn trimesh3_tri2normal(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    tri2nrm: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tri2vtx = crate::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = crate::get_managed_tensor_from_pyany(vtx2xyz)?;
    let tri2nrm = crate::get_managed_tensor_from_pyany(tri2nrm)?;
    let device_type = tri2vtx.ctx.device_type;
    let tri2vtx_sh = unsafe { from_raw_parts(tri2vtx.shape, tri2vtx.ndim as usize) };
    let vtx2xyz_sh = unsafe { from_raw_parts(vtx2xyz.shape, vtx2xyz.ndim as usize) };
    let tri2nrm_sh = unsafe { from_raw_parts(tri2nrm.shape, tri2nrm.ndim as usize) };
    let _num_tri = tri2vtx_sh[0];
    let _num_vtx = vtx2xyz_sh[0];
    assert_eq!(tri2vtx_sh.len(), 2);
    assert_eq!(vtx2xyz_sh.len(), 2);
    assert_eq!(tri2nrm_sh.len(), 2);
    assert_eq!(tri2vtx_sh[1], 3);
    assert_eq!(vtx2xyz_sh[1], 3);
    assert_eq!(tri2nrm_sh[1], 3);
    assert_eq!(tri2vtx_sh, tri2nrm_sh);
    assert_eq!(vtx2xyz.ctx.device_type, device_type);
    assert_eq!(tri2nrm.ctx.device_type, device_type);
    assert_eq!(tri2vtx.dtype.code, dlpack::data_type_codes::UINT);
    assert_eq!(tri2vtx.dtype.bits, 32);
    assert_eq!(vtx2xyz.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(tri2vtx.dtype.bits, 32);
    assert_eq!(tri2nrm.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(tri2nrm.dtype.bits, 32);
    match device_type {
        dlpack::device_type_codes::CPU => {
            let (tri2vtx, _tri2vtx_sh) =
                unsafe { crate::slice_shape_from_tensor::<u32>(tri2vtx).unwrap() };
            let (vtx2xyz, _vtx2xyz_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(vtx2xyz).unwrap() };
            let (tri2nrm, _tri2nrm_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<f32>(tri2nrm).unwrap() };
            del_msh_cpu::trimesh3::tri2normal::<f32, u32>(tri2vtx, vtx2xyz, tri2nrm);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
pub fn trimesh3_bwd_tri2normal(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dw_tri2nrm: &Bound<'_, PyAny>,
    dw_vtx2xyz: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tri2vtx = crate::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = crate::get_managed_tensor_from_pyany(vtx2xyz)?;
    let dw_tri2nrm = crate::get_managed_tensor_from_pyany(dw_tri2nrm)?;
    let dw_vtx2xyz = crate::get_managed_tensor_from_pyany(dw_vtx2xyz)?;
    let device_type = tri2vtx.ctx.device_type;
    let tri2vtx_sh = unsafe { from_raw_parts(tri2vtx.shape, tri2vtx.ndim as usize) };
    let vtx2xyz_sh = unsafe { from_raw_parts(vtx2xyz.shape, vtx2xyz.ndim as usize) };
    let dw_tri2nrm_sh = unsafe { from_raw_parts(dw_tri2nrm.shape, dw_tri2nrm.ndim as usize) };
    let dw_vtx2xyz_sh = unsafe { from_raw_parts(dw_vtx2xyz.shape, dw_vtx2xyz.ndim as usize) };
    let _num_tri = tri2vtx_sh[0];
    let _num_vtx = vtx2xyz_sh[0];
    assert_eq!(tri2vtx_sh.len(), 2);
    assert_eq!(vtx2xyz_sh.len(), 2);
    assert_eq!(dw_tri2nrm_sh.len(), 2);
    assert_eq!(dw_vtx2xyz_sh.len(), 2);
    assert_eq!(tri2vtx_sh[1], 3);
    assert_eq!(vtx2xyz_sh[1], 3);
    assert_eq!(dw_tri2nrm_sh[1], 3);
    assert_eq!(dw_vtx2xyz_sh[1], 3);
    assert_eq!(tri2vtx_sh, dw_tri2nrm_sh);
    assert_eq!(vtx2xyz_sh, dw_vtx2xyz_sh);
    assert_eq!(vtx2xyz.ctx.device_type, device_type);
    assert_eq!(dw_tri2nrm.ctx.device_type, device_type);
    assert_eq!(dw_vtx2xyz.ctx.device_type, device_type);
    assert_eq!(tri2vtx.dtype.code, dlpack::data_type_codes::UINT);
    assert_eq!(tri2vtx.dtype.bits, 32);
    assert_eq!(vtx2xyz.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(tri2vtx.dtype.bits, 32);
    assert_eq!(dw_tri2nrm.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(dw_tri2nrm.dtype.bits, 32);
    assert_eq!(dw_vtx2xyz.dtype.code, dlpack::data_type_codes::FLOAT);
    assert_eq!(dw_vtx2xyz.dtype.bits, 32);
    match device_type {
        dlpack::device_type_codes::CPU => {
            let (tri2vtx, _tri2vtx_sh) =
                unsafe { crate::slice_shape_from_tensor::<u32>(tri2vtx).unwrap() };
            let (vtx2xyz, _vtx2xyz_sh) =
                unsafe { crate::slice_shape_from_tensor::<f32>(vtx2xyz).unwrap() };
            let (dw_tri2nrm, _dw_tri2nrm_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<f32>(dw_tri2nrm).unwrap() };
            let (dw_vtx2xyz, _dw_vtx2xyz_sh) =
                unsafe { crate::slice_shape_from_tensor_mut::<f32>(dw_vtx2xyz).unwrap() };
            del_msh_cpu::trimesh3::bwd_tri2normal::<u32>(tri2vtx, vtx2xyz, dw_tri2nrm, dw_vtx2xyz);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
