use cudarc::driver::{CudaSlice, CudaViewMut, DeviceSlice};
use del_cudarc_safe::cudarc;
use del_cudarc_safe::cudarc::driver::CudaStream;

#[allow(clippy::too_many_arguments)]
pub fn fwd(
    stream: &std::sync::Arc<CudaStream>,
    img_shape: (usize, usize),
    pix2depth: &mut CudaSlice<f32>,
    pix2tri: &CudaSlice<u32>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    transform_ndc2world: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let transform_world2ndc = {
        let transform_ndc2world_cpu = stream.memcpy_dtov(transform_ndc2world)?;
        let transform_ndc2world_cpu = arrayref::array_ref![&transform_ndc2world_cpu, 0, 16];
        let transform_world2ndc_cpu =
            del_geo_core::mat4_col_major::try_inverse(transform_ndc2world_cpu).unwrap();
        stream.memcpy_stod(&transform_world2ndc_cpu)?
    };
    let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let num_tri = tri2vtx.len() / 3;
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "fwd_pix2depth",
        del_msh_cuda_kernel::PIX2DEPTH,
    )?;
    use del_cudarc_safe::cudarc::driver::PushKernelArg;
    {
        let mut builder = stream.launch_builder(&func);
        let num_tri = num_tri as u32;
        let img_width = img_shape.0 as u32;
        let img_height = img_shape.1 as u32;
        builder.arg(pix2depth);
        builder.arg(pix2tri);
        builder.arg(&num_tri);
        builder.arg(tri2vtx);
        builder.arg(vtx2xyz);
        builder.arg(&img_width);
        builder.arg(&img_height);
        builder.arg(transform_ndc2world);
        builder.arg(&transform_world2ndc);
        unsafe { builder.launch(cfg) }?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn bwd_wrt_vtx2xyz(
    stream: &std::sync::Arc<CudaStream>,
    img_shape: (usize, usize),
    dw_vtx2xyz: &mut CudaViewMut<f32>,
    pix2tri: &CudaSlice<u32>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    dw_pix2depth: &CudaSlice<f32>,
    transform_ndc2world: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let transform_world2ndc = {
        let transform_ndc2world_cpu = stream.memcpy_dtov(transform_ndc2world)?;
        let transform_ndc2world_cpu = arrayref::array_ref![&transform_ndc2world_cpu, 0, 16];
        let transform_world2ndc_cpu =
            del_geo_core::mat4_col_major::try_inverse(transform_ndc2world_cpu).unwrap();
        stream.memcpy_stod(&transform_world2ndc_cpu)?
    };
    let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let num_tri = tri2vtx.len() / 3;
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "bwd_wrt_vtx2xyz",
        del_msh_cuda_kernel::PIX2DEPTH,
    )?;
    {
        use del_cudarc_safe::cudarc::driver::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        let img_width = img_shape.0 as u32;
        let img_height = img_shape.1 as u32;
        let num_tri = num_tri as u32;
        builder.arg(dw_vtx2xyz);
        builder.arg(&img_width);
        builder.arg(&img_height);
        builder.arg(pix2tri);
        builder.arg(&num_tri);
        builder.arg(tri2vtx);
        builder.arg(vtx2xyz);
        builder.arg(dw_pix2depth);
        builder.arg(transform_ndc2world);
        builder.arg(&transform_world2ndc);
        unsafe { builder.launch(cfg) }?;
    }
    Ok(())
}
