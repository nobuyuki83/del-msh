use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use del_cudarc_safe::cudarc;

pub fn tri2cntr_from_trimesh3(
    stream: &std::sync::Arc<CudaStream>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    tri2cntr: &mut CudaSlice<f32>,
) -> Result<(), cudarc::driver::DriverError> {
    let num_tri = tri2vtx.len() / 3;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_tri as u32);
    let from_trimsh = del_cudarc_safe::get_or_load_func(
        stream.context(),
        "tri2cntr",
        del_msh_cuda_kernel::BVHNODES_MORTON,
    )?;
    let num_tri = num_tri as u32;
    let mut builder = stream.launch_builder(&from_trimsh);
    builder.arg(tri2cntr);
    builder.arg(&num_tri);
    builder.arg(tri2vtx);
    builder.arg(vtx2xyz);
    unsafe { builder.launch(cfg) }?;
    // let param = (tri2cntr, num_tri as u32, tri2vtx, vtx2xyz);
    //unsafe { from_trimsh.launch(cfg, param) }?;
    Ok(())
}
