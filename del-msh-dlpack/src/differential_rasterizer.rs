use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        differential_rasterizer_antialias,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        differential_rasterizer_bwd_antialias,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn differential_rasterizer_bwd_antialias(
    _py: Python<'_>,
    cedge2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dldw_vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    pix2val: &Bound<'_, PyAny>,
    dldw_pixval: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let cedge2vtx = get_tensor(cedge2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let dldw_vtx2xyz = get_tensor(dldw_vtx2xyz)?;
    let transform_world2pix = get_tensor(transform_world2pix)?;
    let pix2occ = get_tensor(pix2val)?;
    let dldw_pix2occ = get_tensor(dldw_pixval)?;
    let pix2tri = get_tensor(pix2tri)?;
    //
    let num_cedge = shape(&cedge2vtx, 0).unwrap();
    let num_vtx = shape(&vtx2xyz, 0).unwrap();
    let img_h = shape(&dldw_pix2occ, 0).unwrap();
    let img_w = shape(&dldw_pix2occ, 1).unwrap();
    let device = cedge2vtx.ctx.device_type;
    //
    chk2::<u32>(cedge2vtx, num_cedge, 2, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(dldw_vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(dldw_pix2occ, img_h, img_w, device).unwrap();
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::antialias::bwd_antialias(
                slice!(cedge2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice_mut!(dldw_vtx2xyz, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2pix, f32).unwrap(), 0, 16],
                (img_w as usize, img_h as usize),
                slice!(pix2occ, f32).unwrap(),
                slice!(dldw_pix2occ, f32).unwrap(),
                slice!(pix2tri, u32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::differentiable_rasterizer",
                del_msh_cuda_kernels::get("differentiable_rasterizer").unwrap(),
                "antialias_bwd",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_cedge as u32);
            builder.arg_data(&cedge2vtx.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_data(&dldw_vtx2xyz.data);
            builder.arg_u32(img_w as u32);
            builder.arg_u32(img_h as u32);
            builder.arg_data(&dldw_pix2occ.data);
            builder.arg_data(&pix2tri.data);
            builder.arg_data(&transform_world2pix.data);
            const NUM_THREADS_BWD: u32 = 128;
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig {
                        grid_dim: ((num_cedge as u32).div_ceil(NUM_THREADS_BWD), 1, 1),
                        block_dim: (NUM_THREADS_BWD, 1, 1),
                        shared_mem_bytes: 0,
                    },
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
pub fn differential_rasterizer_antialias(
    _py: Python<'_>,
    cedge2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    pic2vin: &Bound<'_, PyAny>,
    pic2vout: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let cedge2vtx = get_tensor(cedge2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let transform_world2pix = get_tensor(transform_world2pix)?;
    let pix2tri = get_tensor(pix2tri)?;
    let pix2vin = get_tensor(pic2vin)?;
    let pix2vout = get_tensor(pic2vout)?;
    //
    let num_cedge = shape(&cedge2vtx, 0).unwrap();
    let num_vtx = shape(&vtx2xyz, 0).unwrap();
    let img_h = shape(&pix2vin, 0).unwrap();
    let img_w = shape(&pix2vin, 1).unwrap();
    let device = cedge2vtx.ctx.device_type;
    //
    chk2::<u32>(cedge2vtx, num_cedge, 2, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    chk2::<f32>(pix2vin, img_h, img_w, device).unwrap();
    chk2::<f32>(pix2vout, img_h, img_w, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::antialias::antialias(
                slice!(cedge2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2pix, f32).unwrap(), 0, 16],
                (img_w as usize, img_h as usize),
                slice!(pix2tri, u32).unwrap(),
                slice!(pix2vin, f32).unwrap(),
                slice_mut!(pix2vout, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::differentiable_rasterizer",
                del_msh_cuda_kernels::get("differentiable_rasterizer").unwrap(),
                "antialias_fwd",
            )
            .unwrap();
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(num_cedge as u32);
            builder.arg_data(&cedge2vtx.data);
            builder.arg_u32(img_w as u32);
            builder.arg_u32(img_h as u32);
            builder.arg_data(&pix2vin.data);
            builder.arg_data(&pix2tri.data);
            builder.arg_data(&vtx2xyz.data);
            builder.arg_data(&transform_world2pix.data);
            const NUM_THREADS_FWD: u32 = 128;
            builder
                .launch_kernel(
                    func,
                    del_cudarc_sys::LaunchConfig {
                        grid_dim: ((num_cedge as u32).div_ceil(NUM_THREADS_FWD), 1, 1),
                        block_dim: (NUM_THREADS_FWD, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
                .unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}
