use candle_core::Tensor;

pub fn contour(
    tri2vtx: &Tensor,
    vtx2xyz: &Tensor,
    transform_world2ndc: &[f32; 16],
) -> anyhow::Result<Tensor> {
    assert_eq!(vtx2xyz.dims2()?.1, 3);
    assert_eq!(tri2vtx.dims2()?.1, 3);
    use std::ops::Deref;
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>(),
        _ => panic!(),
    }?;
    let num_vtx = vtx2xyz.dims2()?.0;
    let vtx2xyz = vtx2xyz.storage_and_layout().0;
    let vtx2xyz = match vtx2xyz.deref() {
        candle_core::Storage::Cpu(cpu_vtx2xyz) => cpu_vtx2xyz.as_slice::<f32>(),
        _ => panic!(),
    }?;
    let edge2vtx = del_msh_core::edge2vtx::from_triangle_mesh(tri2vtx, num_vtx);
    let edge2tri = del_msh_core::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, tri2vtx, num_vtx);
    let edge2vtx_contour = del_msh_core::edge2vtx::contour_for_triangle_mesh(
        tri2vtx,
        vtx2xyz,
        transform_world2ndc,
        &edge2vtx,
        &edge2tri,
    );
    let num_edge = edge2vtx_contour.len() / 2;
    let edge2vtx_contour =
        candle_core::Tensor::from_vec(edge2vtx_contour, (num_edge, 2), &candle_core::Device::Cpu)?;
    Ok(edge2vtx_contour)
}
