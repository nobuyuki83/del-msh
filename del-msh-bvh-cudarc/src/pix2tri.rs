use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

#[allow(clippy::too_many_arguments)]
pub fn pix2tri(
    dev: &std::sync::Arc<CudaDevice>,
    img_shape: (usize, usize),
    pix2tri_dev: &mut CudaSlice<u32>,
    tri2vtx_dev: &CudaSlice<u32>,
    vtx2xyz_dev: &CudaSlice<f32>,
    bvhnodes_dev: &CudaSlice<u32>,
    aabbs_dev: &CudaSlice<f32>,
    transform_ndc2world_dev: &CudaSlice<f32>,
) -> anyhow::Result<()> {
    let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let num_tri = tri2vtx_dev.len() / 3;
    let param = (
        pix2tri_dev,
        num_tri as u32,
        tri2vtx_dev,
        vtx2xyz_dev,
        img_shape.0 as u32,
        img_shape.1 as u32,
        transform_ndc2world_dev,
        bvhnodes_dev,
        aabbs_dev,
    );
    //unsafe { self.pix_to_tri.clone().launch(cfg,param) }.unwrap();
    let pix_to_tri =
        del_cudarc_util::get_or_load_func(dev, "pix_to_tri", del_msh_bvh_cudarc_kernel::PIX2TRI)?;
    use cudarc::driver::LaunchAsync;
    unsafe { pix_to_tri.launch(cfg, param) }?;
    Ok(())
}
