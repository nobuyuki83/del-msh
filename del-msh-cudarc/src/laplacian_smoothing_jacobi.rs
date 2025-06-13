use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, PushKernelArg};
use del_cudarc_safe::cudarc;

pub fn solve(
    stream: &std::sync::Arc<CudaStream>,
    vtx2idx: &CudaSlice<u32>,
    idx2vtx: &CudaSlice<u32>,
    lambda: f32,
    vtx2vars_next: &mut CudaViewMut<f32>,
    vtx2vars_prev: &CudaSlice<f32>,
    vtx2trgs: &CudaSlice<f32>,
) -> Result<(), cudarc::driver::result::DriverError> {
    let num_vtx: u32 = (vtx2idx.len() - 1) as u32;
    let num_dim: u32 = vtx2trgs.len() as u32 / num_vtx;
    assert_eq!((num_vtx * num_dim) as usize, vtx2vars_next.len());
    assert_eq!((num_vtx * num_dim) as usize, vtx2vars_prev.len());
    assert_eq!((num_vtx * num_dim) as usize, vtx2trgs.len());
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx);
    let func = del_cudarc_safe::get_or_load_func(
        &stream.context(),
        "laplacian_smoothing_jacobi",
        del_msh_cuda_kernel::LAPLACIAN_SMOOTHING_JACOBI,
    )?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&num_vtx);
    builder.arg(vtx2idx);
    builder.arg(idx2vtx);
    builder.arg(&lambda);
    builder.arg(vtx2vars_next);
    builder.arg(vtx2vars_prev);
    builder.arg(vtx2trgs);
    unsafe { builder.launch(cfg)? };
    /*
    unsafe { gpu_solve.launch(cfg, param) }?;
    let param = (
        num_vtx,
        vtx2idx,
        idx2vtx,
        lambda,
        vtx2vars_next,
        vtx2vars_prev,
        vtx2trgs,
    );
     */
    Ok(())
}
