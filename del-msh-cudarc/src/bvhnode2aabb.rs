use cudarc::driver::PushKernelArg;
use del_cudarc::cudarc;

pub fn from_trimesh3_with_bvhnodes(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    tri2vtx: &cudarc::driver::CudaSlice<u32>,
    vtx2xyz: &cudarc::driver::CudaSlice<f32>,
    bvhnodes: &cudarc::driver::CudaView<u32>,
    bvhnode2aabb: &mut cudarc::driver::CudaViewMut<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let num_tri = tri2vtx.len() / 3;
    let num_bvhnode = bvhnodes.len() / 3;
    assert_eq!(num_bvhnode, 2 * num_tri - 1);
    let num_branch = num_tri - 1;
    let mut bvhbranch2flag = stream.alloc_zeros::<u32>(num_branch)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let from_trimsh = del_cudarc::get_or_load_func(
        stream.context(),
        "from_trimesh3",
        del_msh_cudarc_kernel::BVHNODE2AABB,
    )?;
    let num_tri = num_tri as u32;
    let mut builder = stream.launch_builder(&from_trimsh);
    builder.arg(bvhnode2aabb);
    builder.arg(&mut bvhbranch2flag);
    builder.arg(&num_bvhnode);
    builder.arg(bvhnodes);
    builder.arg(&num_tri);
    builder.arg(tri2vtx);
    builder.arg(vtx2xyz);
    builder.arg(&(0.));
    unsafe { builder.launch(cfg) }?;
    /*
    let param = (
        bvhnode2aabb,
        &mut bvhbranch2flag,
        num_bvhnode,
        bvhnodes,
        num_tri as u32,
        tri2vtx,
        vtx2xyz,
        0.,
    );
    unsafe { from_trimsh.launch(cfg, param) }?;
     */
    Ok(())
}
