use cudarc::driver::DeviceSlice;
use cudarc::driver::{CudaSlice, CudaViewMut};
use del_cudarc_safe::cudarc;
use del_cudarc_safe::cudarc::driver::{CudaStream, PushKernelArg};

pub fn compute_with_alias(
    stream: &std::sync::Arc<CudaStream>,
    img_shape: (usize, usize),
    pix2tri: &CudaSlice<u32>,
) -> Result<CudaSlice<f32>, cudarc::driver::DriverError> {
    let mut img = stream.alloc_zeros::<f32>(img_shape.1 * img_shape.0)?;
    del_cudarc_safe::util::set_value_at_mask(stream, &mut img, 1f32, pix2tri, u32::MAX, false)?;
    Ok(img)
}

pub fn remove_alias(
    stream: &std::sync::Arc<CudaStream>,
    edge2vtx_contour: &CudaSlice<u32>,
    img_shape: (usize, usize),
    pix2occu: &mut CudaSlice<f32>,
    pix2tri: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    transform_world2pix: &CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let num_edge = edge2vtx_contour.len() / 2;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_edge as u32);
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "silhouette_fwd",
        del_msh_cuda_kernel::SILHOUETTE,
    )?;
    use del_cudarc_safe::cudarc::driver::PushKernelArg;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&(num_edge as u32));
    builder.arg(edge2vtx_contour);
    builder.arg(&(img_shape.0 as u32));
    builder.arg(&(img_shape.1 as u32));
    builder.arg(pix2occu);
    builder.arg(pix2tri);
    builder.arg(vtx2xyz);
    builder.arg(transform_world2pix);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

pub fn backward_wrt_vtx2xyz(
    stream: &std::sync::Arc<CudaStream>,
    edge2vtx_contour: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    dldw_vtx2xyz: &mut CudaViewMut<f32>,
    transform_world2pix: &CudaSlice<f32>,
    img_shape: (usize, usize),
    dldw_pix2occl: &CudaSlice<f32>,
    pix2tri: &CudaSlice<u32>,
) -> Result<(), cudarc::driver::DriverError> {
    let num_edge = edge2vtx_contour.len() / 2;
    let cfg = {
        let num_threads = 512u32;
        let num_blocks = (num_edge as u32).div_ceil(num_threads);
        cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (num_threads, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "silhouette_bwd",
        del_msh_cuda_kernel::SILHOUETTE,
    )?;
    use del_cudarc_safe::cudarc::driver::PushKernelArg;
    let mut builder = stream.launch_builder(&func);
    builder.arg(num_edge as u32);
    builder.arg(edge2vtx_contour);
    builder.arg(vtx2xyz);
    builder.arg(dldw_vtx2xyz);
    builder.arg(&(img_shape.0 as u32));
    builder.arg(&(img_shape.1 as u32));
    builder.arg(dldw_pix2occl);
    builder.arg(pix2tri);
    builder.arg(transform_world2pix);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}
