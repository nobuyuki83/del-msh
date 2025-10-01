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
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let tri2vtx = crate::get_managed_tensor_from_pyany(tri2vtx)?;
    let vtx2xyz = crate::get_managed_tensor_from_pyany(vtx2xyz)?;
    let tri2nrm = crate::get_managed_tensor_from_pyany(tri2nrm)?;
    let device_type = tri2vtx.ctx.device_type;
    let tri2vtx_sh = unsafe { from_raw_parts(tri2vtx.shape, tri2vtx.ndim as usize) };
    let vtx2xyz_sh = unsafe { from_raw_parts(vtx2xyz.shape, vtx2xyz.ndim as usize) };
    let tri2nrm_sh = unsafe { from_raw_parts(tri2nrm.shape, tri2nrm.ndim as usize) };
    let num_tri = tri2vtx_sh[0];
    let num_vtx = vtx2xyz_sh[0];
    //
    assert_eq!(tri2vtx.byte_offset, 0);
    assert_eq!(vtx2xyz.byte_offset, 0);
    assert_eq!(tri2nrm.byte_offset, 0);
    assert_eq!(tri2vtx_sh, vec!(num_tri, 3));
    assert_eq!(vtx2xyz_sh, vec!(num_vtx, 3));
    assert_eq!(tri2nrm_sh, vec!(num_tri, 3));
    assert!(crate::is_equal::<i32>(&tri2vtx.dtype));
    assert!(crate::is_equal::<f32>(&vtx2xyz.dtype));
    assert!(crate::is_equal::<f32>(&tri2nrm.dtype));
    assert!(unsafe { crate::is_tensor_c_contiguous(tri2vtx) });
    assert!(unsafe { crate::is_tensor_c_contiguous(vtx2xyz) });
    assert!(unsafe { crate::is_tensor_c_contiguous(tri2nrm) });
    assert_eq!(vtx2xyz.ctx.device_type, device_type);
    assert_eq!(tri2nrm.ctx.device_type, device_type);
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            let tri2vtx = unsafe { crate::slice_from_tensor::<i32>(tri2vtx).unwrap() };
            assert_eq!(tri2vtx_sh, vec!(num_tri, 3));
            let vtx2xyz = unsafe { crate::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            let tri2nrm = unsafe { crate::slice_from_tensor_mut::<f32>(tri2nrm).unwrap() };
            del_msh_cpu::trimesh3::tri2normal::<f32, i32>(tri2vtx, vtx2xyz, tri2nrm);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            //println!("GPU_{}", tri2vtx.ctx.device_id);
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::TRIMESH3,
                "tri2normal",
            );
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            assert_eq!(size_of::<usize>(), 8);
            let stream = stream_ptr as usize as *mut std::ffi::c_void as cu::CUstream;
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_i32(num_tri as i32);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_data(&tri2nrm.data);
            builder.launch_kernel(function, num_tri as u32);
        }
        _ => {
            println!("Unknown device type {}", device_type);
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
    #[allow(unused_variables)] stream_ptr: u64,
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
    let num_tri = tri2vtx_sh[0];
    let num_vtx = vtx2xyz_sh[0];
    //
    assert_eq!(tri2vtx.byte_offset, 0);
    assert_eq!(vtx2xyz.byte_offset, 0);
    assert_eq!(dw_tri2nrm.byte_offset, 0);
    assert_eq!(dw_vtx2xyz.byte_offset, 0);
    assert_eq!(tri2vtx_sh, vec!(num_tri, 3));
    assert_eq!(vtx2xyz_sh, vec!(num_vtx, 3));
    assert_eq!(dw_tri2nrm_sh, vec!(num_tri, 3));
    assert_eq!(dw_vtx2xyz_sh, vec!(num_vtx, 3));
    assert_eq!(tri2vtx_sh, dw_tri2nrm_sh);
    assert_eq!(vtx2xyz_sh, dw_vtx2xyz_sh);
    assert!(crate::is_equal::<i32>(&tri2vtx.dtype));
    assert!(crate::is_equal::<f32>(&vtx2xyz.dtype));
    assert!(crate::is_equal::<f32>(&dw_tri2nrm.dtype));
    assert!(crate::is_equal::<f32>(&dw_vtx2xyz.dtype));
    assert!(unsafe { crate::is_tensor_c_contiguous(tri2vtx) });
    assert!(unsafe { crate::is_tensor_c_contiguous(vtx2xyz) });
    assert!(unsafe { crate::is_tensor_c_contiguous(dw_tri2nrm) });
    assert!(unsafe { crate::is_tensor_c_contiguous(dw_vtx2xyz) });
    assert_eq!(vtx2xyz.ctx.device_type, device_type);
    assert_eq!(dw_tri2nrm.ctx.device_type, device_type);
    assert_eq!(dw_vtx2xyz.ctx.device_type, device_type);
    //
    match device_type {
        dlpack::device_type_codes::CPU => {
            let tri2vtx = unsafe { crate::slice_from_tensor::<i32>(tri2vtx).unwrap() };
            assert_eq!(tri2vtx_sh, vec!(num_tri, 3));
            let vtx2xyz = unsafe { crate::slice_from_tensor::<f32>(vtx2xyz).unwrap() };
            assert_eq!(vtx2xyz_sh, vec!(num_vtx, 3));
            let dw_tri2nrm = unsafe { crate::slice_from_tensor_mut::<f32>(dw_tri2nrm).unwrap() };
            let dw_vtx2xyz = unsafe { crate::slice_from_tensor_mut::<f32>(dw_vtx2xyz).unwrap() };
            del_msh_cpu::trimesh3::bwd_tri2normal::<i32>(tri2vtx, vtx2xyz, dw_tri2nrm, dw_vtx2xyz);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            // println!("GPU_{}", tri2vtx.ctx.device_id);
            let (function, _module) = del_cudarc_sys::load_function_in_module(
                del_msh_cuda_kernel::TRIMESH3,
                "bwd_tri2normal",
            );
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            assert_eq!(size_of::<usize>(), 8);
            let stream = stream_ptr as usize as *mut std::ffi::c_void as cu::CUstream;
            let dptr: cu::CUdeviceptr = dw_vtx2xyz.data as cu::CUdeviceptr;
            cuda_check!(cu::cuMemsetD32Async(
                dptr,
                0,
                (num_vtx * 3) as usize,
                stream
            ));
            cuda_check!(cu::cuStreamSynchronize(stream));
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_i32(num_tri as i32);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_data(&dw_tri2nrm.data);
            builder.arg_data(&dw_vtx2xyz.data);
            builder.launch_kernel(function, num_tri as u32);
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
