use std::ops::Deref;

pub fn from_trimesh2(
    tri2vtx: &candle_core::Tensor,
    vtx2xy: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
        _ => panic!(),
    };
    let vtx2xy = vtx2xy.storage_and_layout().0;
    let vtx2xy = match vtx2xy.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh::<u32>(tri2vtx, vtx2xy, 2);
    let aabbs = del_msh_core::bvhnode2aabb2::from_uniform_mesh_with_bvh::<u32, f32>(
        0,
        &bvhnodes,
        Some((tri2vtx, 3)),
        vtx2xy,
        None,
    );
    let num_bvhnode = bvhnodes.len() / 3;
    let bvhnodes = candle_core::Tensor::from_vec(
        bvhnodes,
        candle_core::Shape::from((num_bvhnode, 3)),
        &candle_core::Device::Cpu,
    )?;
    let num_aabb = aabbs.len() / 4;
    let aabbs = candle_core::Tensor::from_vec(
        aabbs,
        candle_core::Shape::from((num_aabb, 4)),
        &candle_core::Device::Cpu,
    )?;
    Ok((bvhnodes, aabbs))
}

pub fn from_trimesh3(
    tri2vtx: &candle_core::Tensor,
    vtx2xyz: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
        _ => panic!(),
    };
    let vtx2xyz = vtx2xyz.storage_and_layout().0;
    let vtx2xyz = match vtx2xyz.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh::<u32>(tri2vtx, vtx2xyz, 3);
    let aabbs = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh::<u32, f32>(
        0,
        &bvhnodes,
        Some((tri2vtx, 3)),
        vtx2xyz,
        None,
    );
    let num_bvhnode = bvhnodes.len() / 3;
    let bvhnodes = candle_core::Tensor::from_vec(
        bvhnodes,
        candle_core::Shape::from((num_bvhnode, 3)),
        &candle_core::Device::Cpu,
    )?;
    let num_aabb = aabbs.len() / 6;
    let aabbs = candle_core::Tensor::from_vec(
        aabbs,
        candle_core::Shape::from((num_aabb, 6)),
        &candle_core::Device::Cpu,
    )?;
    Ok((bvhnodes, aabbs))
}
