use cudarc::driver::{CudaSlice, PushKernelArg};
use del_cudarc::cudarc;

pub fn fwd(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    edge2vtx: &CudaSlice<u32>,
    edge2tri: &CudaSlice<u32>,
    transform_world2ndc: &cudarc::driver::CudaSlice<f32>,
) -> Result<cudarc::driver::CudaSlice<u32>, cudarc::driver::DriverError> {
    let num_edge = edge2vtx.len() / 2;
    let transform_ndc2world = {
        let transform_world2ndc_cpu = stream.memcpy_dtov(transform_world2ndc)?;
        let transform_world2ndc_cpu = arrayref::array_ref![&transform_world2ndc_cpu, 0, 16];
        let transform_ndc2world_cpu =
            del_geo_core::mat4_col_major::try_inverse(transform_world2ndc_cpu).unwrap();
        stream.memcpy_stod(&transform_ndc2world_cpu)?
    };
    let edge2flg = {
        let mut edge2flg = stream.alloc_zeros::<u32>(num_edge + 1)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_edge as u32);
        let func = del_cudarc::get_or_load_func(
            stream.context(),
            "edge2vtx_contour_set_flag",
            del_msh_cudarc_kernel::EDGE2VTX,
        )?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&num_edge);
        builder.arg(&mut edge2flg);
        builder.arg(edge2vtx);
        builder.arg(edge2tri);
        builder.arg(tri2vtx);
        builder.arg(vtx2xyz);
        builder.arg(transform_world2ndc);
        builder.arg(&transform_ndc2world);
        unsafe { builder.launch(cfg) }?;
        /*
        unsafe { func.launch(cfg, param) }?;
        let param = (
            num_edge,
            &mut edge2flg,
            edge2vtx,
            edge2tri,
            tri2vtx,
            vtx2xyz,
            transform_world2ndc,
            &transform_ndc2world,
        );
         */
        edge2flg
    };
    let edge2vtx_contour =
        del_cudarc::get_flagged_element::get_flagged_element(stream, edge2vtx, &edge2flg)?;
    Ok(edge2vtx_contour)
}
