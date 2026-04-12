use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(pix2depth_update, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pix2depth_bwd_wrt_vtx2xyz, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn pix2depth_update(
    _py: Python<'_>,
    pix2depth: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_ndc2world: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let pix2depth = get_tensor(pix2depth)?;
    let pix2tri = get_tensor(pix2tri)?;
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let transform_ndc2world = get_tensor(transform_ndc2world)?;
    //
    let device = pix2depth.ctx.device_type;
    let img_shape = [shape(&pix2depth, 1).unwrap(), shape(&pix2depth, 0).unwrap()];
    let num_tri = shape(&tri2vtx, 0).unwrap();
    let num_vtx = shape(&vtx2xyz, 0).unwrap();
    //
    chk2::<f32>(&pix2depth, img_shape[1], img_shape[0], device).unwrap();
    chk2::<u32>(&pix2tri, img_shape[1], img_shape[0], device).unwrap();
    chk2::<u32>(&tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(&vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(&transform_ndc2world, 4, 4, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::trimesh3_raycast::pix2depth_from_pix2tri(
                slice_mut!(pix2depth, f32).unwrap(),
                slice!(pix2tri, u32).unwrap(),
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                (img_shape[0] as usize, img_shape[1] as usize),
                arrayref::array_ref![slice!(transform_ndc2world, f32).unwrap(), 0, 16],
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::pix2depth",
                del_msh_cuda_kernels::get("pix2depth").unwrap(),
                "fwd",
            )
            .unwrap();
            let transform_world2ndc = {
                let transform_ndc2world = del_cudarc_sys::CuVec::<f32>::from_dptr(transform_ndc2world.data as del_cudarc_sys::cu::CUdeviceptr, 16);
                let transform_ndc2world = transform_ndc2world.copy_to_host().unwrap();
                let ndc2world = arrayref::array_ref![&transform_ndc2world, 0, 16];
                del_geo_core::mat4_col_major::try_inverse(ndc2world).unwrap()
            };
            let transform_world2ndc =
                del_cudarc_sys::CuVec::<f32>::from_slice(&transform_world2ndc).unwrap();
            let num_pix = (img_shape[0] * img_shape[1]) as usize;
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_data(&pix2depth.data);
            builder.arg_data(&pix2tri.data);
            builder.arg_u32(num_tri as u32);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_u32(img_shape[0] as u32);
            builder.arg_u32(img_shape[1] as u32);
            builder.arg_data(&transform_ndc2world.data);
            builder.arg_dptr(transform_world2ndc.dptr);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pix as u32),
                )
                .unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
pub fn pix2depth_bwd_wrt_vtx2xyz(
    _py: Python<'_>,
    dldw_vtx2xyz: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dldw_pix2depth: &Bound<'_, PyAny>,
    transform_ndc2world: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let dldw_vtx2xyz = get_tensor(dldw_vtx2xyz)?;
    let pix2tri = get_tensor(pix2tri)?;
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let dldw_pix2depth = get_tensor(dldw_pix2depth)?;
    let transform_ndc2world = get_tensor(transform_ndc2world)?;
    //
    let device = dldw_vtx2xyz.ctx.device_type;
    let num_vtx = shape(&vtx2xyz, 0).unwrap();
    let num_tri = shape(&tri2vtx, 0).unwrap();
    let img_shape = [shape(&pix2tri, 1).unwrap(), shape(&pix2tri, 0).unwrap()];
    //
    chk2::<f32>(&dldw_vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<u32>(&pix2tri, img_shape[1], img_shape[0], device).unwrap();
    chk2::<u32>(&tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(&vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(&dldw_pix2depth, img_shape[1], img_shape[0], device).unwrap();
    chk2::<f32>(&transform_ndc2world, 4, 4, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::trimesh3_raycast::bwd_continuous(
                slice!(pix2tri, u32).unwrap(),
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice!(dldw_pix2depth, f32).unwrap(),
                arrayref::array_ref![slice!(transform_ndc2world, f32).unwrap(), 0, 16],
                (img_shape[0] as usize, img_shape[1] as usize),
                slice_mut!(dldw_vtx2xyz, f32).unwrap(),
                &del_msh_cpu::trimesh3_raycast::Depth,
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::pix2depth",
                del_msh_cuda_kernels::get("pix2depth").unwrap(),
                "bwd_wrt_vtx2xyz",
            )
            .unwrap();
            let num_pix = (img_shape[0] * img_shape[1]) as usize;
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_data(&dldw_vtx2xyz.data);
            builder.arg_data(&pix2tri.data);
            builder.arg_data(&tri2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_data(&dldw_pix2depth.data);
            builder.arg_u32(img_shape[0] as u32);
            builder.arg_u32(img_shape[1] as u32);
            builder.arg_data(&transform_ndc2world.data);
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig::for_num_elems(num_pix as u32),
                )
                .unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
