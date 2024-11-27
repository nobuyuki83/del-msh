#[cfg(feature = "cuda")]
pub fn test(
    dev: &candle_core::Device,
    tri2vtx: &candle_core::Tensor,
    vtx2xyz: &candle_core::Tensor,
) -> anyhow::Result<()> {
    let dev = dev.as_cuda_device()?;
    use std::ops::Deref;
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cuda(cpu_tri2vtx) => cpu_tri2vtx.as_cuda_slice::<u32>(),
        _ => panic!(),
    }?;
    let vtx2xyz = vtx2xyz.storage_and_layout().0;
    let vtx2xyz = match vtx2xyz.deref() {
        candle_core::Storage::Cuda(cpu_vtx2xyz) => cpu_vtx2xyz.as_cuda_slice::<f32>(),
        _ => panic!(),
    }?;
    // let mut tri2cntr = dev.alloc_zeros::<f32>(num_tri*3)?;
    // del_cudarc_bvh::bvh::tri2cntr_from_trimesh3(dev, tri2vtx, vtx2xyz, &mut tri2cntr)?;
    let now = std::time::Instant::now();
    let (bvhnodes, bvhnode2aabb) = del_msh_cudarc::make_bvh_from_trimesh3(dev, tri2vtx, vtx2xyz)?;
    println!("{:?}", now.elapsed());
    Ok(())
}
