use del_dlpack::{
    check_1d_tensor as chk1, check_2d_tensor as chk2, check_3d_tensor as chk3, dlpack,
    get_managed_tensor_from_pyany as get_tensor, get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(rasterized_edge_gradient_bwd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        rasterized_edge_gradient_edge_gradient_and_type,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        rasterized_edge_gradient_smooth_gradient,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        rasterized_edge_gradient_interpolate,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_bwd(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    dldw_vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    pix2val: &Bound<'_, PyAny>,
    dldw_pix2val: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let dldw_vtx2xyz = get_tensor(dldw_vtx2xyz)?;
    let transform_world2pix = get_tensor(transform_world2pix)?;
    let pix2tri = get_tensor(pix2tri)?;
    let pix2val = get_tensor(pix2val)?;
    let dldw_pix2val = get_tensor(dldw_pix2val)?;
    //
    let device = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let img_h = shape(pix2tri, 0).unwrap();
    let img_w = shape(pix2tri, 1).unwrap();
    let num_vdim = shape(pix2val, 2).unwrap();
    //
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk2::<f32>(dldw_vtx2xyz, num_vtx, 3, device).unwrap();
    chk1::<f32>(transform_world2pix, 16, device).unwrap();
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    chk3::<f32>(pix2val, img_h, img_w, num_vdim, device).unwrap();
    chk3::<f32>(dldw_pix2val, img_h, img_w, num_vdim, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::bwd(
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                slice_mut!(dldw_vtx2xyz, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2pix, f32).unwrap(), 0, 16],
                (img_w as usize, img_h as usize),
                slice!(pix2tri, u32).unwrap(),
                num_vdim as usize,
                slice!(dldw_pix2val, f32).unwrap(),
                slice!(pix2val, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            const BLOCK: u32 = 16;
            // hedge: grid covers img_w × (img_h - 1)
            {
                let func = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::rasterized_edge_gradient",
                    del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                    "bwd_hedge",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig {
                    grid_dim: (
                        (img_w as u32).div_ceil(BLOCK),
                        ((img_h - 1) as u32).div_ceil(BLOCK),
                        1,
                    ),
                    block_dim: (BLOCK, BLOCK, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_data(&tri2vtx.data);
                builder.arg_data(&vtx2xyz.data);
                builder.arg_data(&dldw_vtx2xyz.data);
                builder.arg_data(&transform_world2pix.data);
                builder.arg_u32(img_w as u32);
                builder.arg_u32(img_h as u32);
                builder.arg_data(&pix2tri.data);
                builder.arg_u32(num_vdim as u32);
                builder.arg_data(&dldw_pix2val.data); // swapped: mirrors CPU call order
                builder.arg_data(&pix2val.data);
                builder.launch_kernel(func, cfg).unwrap();
            }
            // vedge: grid covers (img_w - 1) × img_h
            {
                let func = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::rasterized_edge_gradient",
                    del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                    "bwd_vedge",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig {
                    grid_dim: (
                        ((img_w - 1) as u32).div_ceil(BLOCK),
                        (img_h as u32).div_ceil(BLOCK),
                        1,
                    ),
                    block_dim: (BLOCK, BLOCK, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_data(&tri2vtx.data);
                builder.arg_data(&vtx2xyz.data);
                builder.arg_data(&dldw_vtx2xyz.data);
                builder.arg_data(&transform_world2pix.data);
                builder.arg_u32(img_w as u32);
                builder.arg_u32(img_h as u32);
                builder.arg_data(&pix2tri.data);
                builder.arg_u32(num_vdim as u32);
                builder.arg_data(&dldw_pix2val.data); // swapped: mirrors CPU call order
                builder.arg_data(&pix2val.data);
                builder.launch_kernel(func, cfg).unwrap();
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "unsupported device: {}",
                del_dlpack::device_type_code_to_str(device)
            )));
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_edge_gradient_and_type(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    transform_world2pix: &Bound<'_, PyAny>,
    pix2tri: &Bound<'_, PyAny>,
    pix2val: &Bound<'_, PyAny>,
    dldw_pixval: &Bound<'_, PyAny>,
    hedge2type: &Bound<'_, PyAny>,
    hedge2dldr: &Bound<'_, PyAny>,
    vedge2type: &Bound<'_, PyAny>,
    vedge2dldr: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let transform_world2pix = get_tensor(transform_world2pix)?;
    let pix2tri = get_tensor(pix2tri)?;
    let pix2val = get_tensor(pix2val)?;
    let dldw_pixval = get_tensor(dldw_pixval)?;
    let hedge2type = get_tensor(hedge2type)?;
    let hedge2dldr = get_tensor(hedge2dldr)?;
    let vedge2type = get_tensor(vedge2type)?;
    let vedge2dldr = get_tensor(vedge2dldr)?;
    //
    let device = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_vdim = shape(dldw_pixval, 2).unwrap();
    let img_h = shape(pix2tri, 0).unwrap();
    let img_w = shape(pix2tri, 1).unwrap();
    //
    chk2::<u32>(tri2vtx, num_tri, 3, device).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device).unwrap();
    chk1::<f32>(transform_world2pix, 16, device).unwrap();
    chk2::<u32>(pix2tri, img_h, img_w, device).unwrap();
    chk3::<f32>(pix2val, img_h, img_w, num_vdim, device).unwrap();
    chk3::<f32>(dldw_pixval, img_h, img_w, num_vdim, device).unwrap();
    chk2::<u8>(hedge2type, img_h - 1, img_w, device).unwrap();
    chk2::<f32>(hedge2dldr, img_h - 1, img_w, device).unwrap();
    chk2::<u8>(vedge2type, img_h, img_w - 1, device).unwrap();
    chk2::<f32>(vedge2dldr, img_h, img_w - 1, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::edge_gradient_and_type(
                slice!(tri2vtx, u32).unwrap(),
                slice!(vtx2xyz, f32).unwrap(),
                arrayref::array_ref![slice!(transform_world2pix, f32).unwrap(), 0, 16],
                (img_w as usize, img_h as usize),
                slice!(pix2tri, u32).unwrap(),
                num_vdim as usize,
                slice!(pix2val, f32).unwrap(),
                slice!(dldw_pixval, f32).unwrap(),
                slice_mut!(hedge2type, u8).unwrap(),
                slice_mut!(hedge2dldr, f32).unwrap(),
                slice_mut!(vedge2type, u8).unwrap(),
                slice_mut!(vedge2dldr, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            const BLOCK: u32 = 16;
            // horizontal edges: img_w * (img_h - 1) work items
            {
                let func = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::rasterized_edge_gradient",
                    del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                    "hedge_gradient_and_type",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig {
                    grid_dim: (
                        (img_w as u32).div_ceil(BLOCK),
                        ((img_h - 1) as u32).div_ceil(BLOCK),
                        1,
                    ),
                    block_dim: (BLOCK, BLOCK, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_data(&tri2vtx.data);
                builder.arg_data(&vtx2xyz.data);
                builder.arg_data(&transform_world2pix.data);
                builder.arg_u32(img_w as u32);
                builder.arg_u32(img_h as u32);
                builder.arg_data(&pix2tri.data);
                builder.arg_u32(num_vdim as u32);
                builder.arg_data(&pix2val.data);
                builder.arg_data(&dldw_pixval.data);
                builder.arg_data(&hedge2type.data);
                builder.arg_data(&hedge2dldr.data);
                builder.launch_kernel(func, cfg).unwrap();
            }
            // vertical edges: (img_w - 1) * img_h work items
            {
                let func = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::rasterized_edge_gradient",
                    del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                    "vedge_gradient_and_type",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig {
                    grid_dim: (
                        ((img_w - 1) as u32).div_ceil(BLOCK),
                        (img_h as u32).div_ceil(BLOCK),
                        1,
                    ),
                    block_dim: (BLOCK, BLOCK, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = del_cudarc_sys::Builder::new(stream);
                builder.arg_data(&tri2vtx.data);
                builder.arg_data(&vtx2xyz.data);
                builder.arg_data(&transform_world2pix.data);
                builder.arg_u32(img_w as u32);
                builder.arg_u32(img_h as u32);
                builder.arg_data(&pix2tri.data);
                builder.arg_u32(num_vdim as u32);
                builder.arg_data(&pix2val.data);
                builder.arg_data(&dldw_pixval.data);
                builder.arg_data(&vedge2type.data);
                builder.arg_data(&vedge2dldr.data);
                builder.launch_kernel(func, cfg).unwrap();
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "device not supported",
            ));
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_smooth_gradient(
    _py: Python<'_>,
    hedge2type: &Bound<'_, PyAny>,
    hedge2dldr: &Bound<'_, PyAny>,
    vedge2type: &Bound<'_, PyAny>,
    vedge2dldr: &Bound<'_, PyAny>,
    num_iter: usize,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let hedge2type = get_tensor(hedge2type)?;
    let hedge2dldr = get_tensor(hedge2dldr)?;
    let vedge2type = get_tensor(vedge2type)?;
    let vedge2dldr = get_tensor(vedge2dldr)?;
    //
    let device = hedge2type.ctx.device_type;
    let img_h_m1 = shape(hedge2type, 0).unwrap(); // H-1
    let img_w = shape(hedge2type, 1).unwrap(); // W
    let img_h = img_h_m1 + 1;
    //
    chk2::<u8>(hedge2type, img_h_m1, img_w, device).unwrap();
    chk2::<f32>(hedge2dldr, img_h_m1, img_w, device).unwrap();
    chk2::<u8>(vedge2type, img_h, img_w - 1, device).unwrap();
    chk2::<f32>(vedge2dldr, img_h, img_w - 1, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::smooth_gradient_hedge(
                (img_w as usize, img_h as usize),
                slice!(hedge2type, u8).unwrap(),
                slice!(vedge2type, u8).unwrap(),
                num_iter,
                slice_mut!(hedge2dldr, f32).unwrap(),
            );
            del_msh_cpu::rasterized_edge_gradient::smooth_gradient_vedge(
                (img_w as usize, img_h as usize),
                slice!(hedge2type, u8).unwrap(),
                slice!(vedge2type, u8).unwrap(),
                num_iter,
                slice_mut!(vedge2dldr, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            {
                let func = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::rasterized_edge_gradient",
                    del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                    "smooth_hedge_red_black",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(((img_h - 1) * img_w) as u32);
                for _ in 0..num_iter {
                    for color in [0u32, 1u32] {
                        let mut builder = del_cudarc_sys::Builder::new(stream);
                        builder.arg_u32(img_w as u32);
                        builder.arg_u32(img_h as u32);
                        builder.arg_data(&hedge2type.data);
                        builder.arg_data(&hedge2dldr.data);
                        builder.arg_data(&vedge2type.data);
                        builder.arg_u32(color);
                        builder.launch_kernel(func, cfg).unwrap();
                    }
                }
            }
            {
                let func = del_cudarc_sys::cache_func::get_function_cached(
                    "del_msh::rasterized_edge_gradient",
                    del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                    "smooth_vedge_red_black",
                )
                .unwrap();
                let cfg = del_cudarc_sys::LaunchConfig::for_num_elems((img_h * (img_w - 1)) as u32);
                for _ in 0..num_iter {
                    for color in [0u32, 1u32] {
                        let mut builder = del_cudarc_sys::Builder::new(stream);
                        builder.arg_u32(img_w as u32);
                        builder.arg_u32(img_h as u32);
                        builder.arg_data(&vedge2type.data);
                        builder.arg_data(&vedge2dldr.data);
                        builder.arg_data(&hedge2type.data);
                        builder.arg_u32(color);
                        builder.launch_kernel(func, cfg).unwrap();
                    }
                }
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported",
            ));
        }
    }
    Ok(())
}

#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rasterized_edge_gradient_interpolate(
    _py: Python<'_>,
    hedge2vy: &Bound<'_, PyAny>,
    vedge2vx: &Bound<'_, PyAny>,
    vtx2xy: &Bound<'_, PyAny>,
    vtx2velo: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let hedge2vy = get_tensor(hedge2vy)?;
    let vedge2vx = get_tensor(vedge2vx)?;
    let vtx2xy = get_tensor(vtx2xy)?;
    let vtx2velo = get_tensor(vtx2velo)?;
    //
    let device = hedge2vy.ctx.device_type;
    let img_h = shape(hedge2vy, 0).unwrap() + 1; // H
    let img_w = shape(hedge2vy, 1).unwrap(); // W
    let num_vtx = shape(vtx2velo, 0).unwrap();
    //
    chk2::<f32>(hedge2vy, img_h - 1, img_w, device).unwrap();
    chk2::<f32>(vedge2vx, img_h, img_w - 1, device).unwrap();
    chk2::<f32>(vtx2xy, num_vtx, 2, device).unwrap();
    chk2::<f32>(vtx2velo, num_vtx, 2, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            del_msh_cpu::rasterized_edge_gradient::interpolate_staggered_grid(
                (img_w as usize, img_h as usize),
                slice_mut!(hedge2vy, f32).unwrap(),
                slice_mut!(vedge2vx, f32).unwrap(),
                slice!(vtx2xy, f32).unwrap(),
                slice_mut!(vtx2velo, f32).unwrap(),
            );
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let func = del_cudarc_sys::cache_func::get_function_cached(
                "del_msh::rasterized_edge_gradient",
                del_msh_cuda_kernels::get("rasterized_edge_gradient").unwrap(),
                "interpolate_staggered_grid",
            )
            .unwrap();
            let cfg = del_cudarc_sys::LaunchConfig::for_num_elems(num_vtx as u32);
            let mut builder = del_cudarc_sys::Builder::new(stream);
            builder.arg_u32(img_w as u32);
            builder.arg_u32(img_h as u32);
            builder.arg_data(&hedge2vy.data);
            builder.arg_data(&vedge2vx.data);
            builder.arg_data(&vtx2xy.data);
            builder.arg_data(&vtx2velo.data);
            builder.arg_u32(num_vtx as u32);
            builder.launch_kernel(func, cfg).unwrap();
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "GPU not supported",
            ));
        }
    }
    Ok(())
}
