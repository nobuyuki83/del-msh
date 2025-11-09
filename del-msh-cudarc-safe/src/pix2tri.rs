use cudarc::driver::CudaSlice;
use del_cudarc_safe::cudarc;
use del_cudarc_safe::cudarc::driver::CudaStream;

#[allow(clippy::too_many_arguments)]
pub fn fwd(
    stream: &std::sync::Arc<CudaStream>,
    img_shape: (usize, usize),
    pix2tri: &mut CudaSlice<u32>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    bvhnodes: &CudaSlice<u32>,
    bvhnode2aabb: &CudaSlice<f32>,
    transform_ndc2world: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    // let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let cfg = {
        let n = (img_shape.0 * img_shape.1) as u32;
        const NUM_THREADS: u32 = 32;
        let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let num_tri = tri2vtx.len() / 3;
    let func = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "pix_to_tri",
        del_msh_cuda_kernel::PIX2TRI,
    )?;
    {
        use del_cudarc_safe::cudarc::driver::PushKernelArg;
        let mut builder = stream.launch_builder(&func);
        let num_tri = num_tri as u32;
        let img_width = img_shape.0 as u32;
        let img_height = img_shape.1 as u32;
        builder.arg(pix2tri);
        builder.arg(&num_tri);
        builder.arg(tri2vtx);
        builder.arg(vtx2xyz);
        builder.arg(&img_width);
        builder.arg(&img_height);
        builder.arg(transform_ndc2world);
        builder.arg(bvhnodes);
        builder.arg(bvhnode2aabb);
        unsafe { builder.launch(cfg) }?;
    }
    Ok(())
}
