use candle_core::Tensor;
use std::ops::Deref;

pub fn raycast2(
    tri2vtx: &Tensor,
    vtx2xy: &Tensor,
    bvhnodes: &Tensor,
    bvhnode2aabb: &Tensor,
    img_shape: &(usize, usize),  // (width, height)
    transform_xy2pix: &[f32; 9], // transform column major
) -> candle_core::Result<Tensor> {
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
        _ => panic!(),
    };
    let (_num_vtx, two) = vtx2xy.shape().dims2()?;
    assert_eq!(two, 2);
    let vtx2xy = vtx2xy.storage_and_layout().0;
    let vtx2xy = match vtx2xy.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let aabbs = bvhnode2aabb.storage_and_layout().0;
    let aabbs = match aabbs.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let bvhnodes = bvhnodes.storage_and_layout().0;
    let bvhnodes = match bvhnodes.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
        _ => panic!(),
    };
    let mut img = vec![u32::MAX; img_shape.0 * img_shape.1];
    let transform_pix2xy = del_geo_core::mat3_col_major::try_inverse(transform_xy2pix).unwrap();
    for i_h in 0..img_shape.1 {
        for i_w in 0..img_shape.0 {
            let p_xy = del_geo_core::mat3_col_major::transform_homogeneous(
                &transform_pix2xy,
                &[i_w as f32 + 0.5, i_h as f32 + 0.5],
            )
            .unwrap();
            let mut res: Vec<(u32, f32, f32)> = vec![];
            del_msh_cpu::search_bvh2::including_point::<f32, u32>(
                &mut res, tri2vtx, vtx2xy, &p_xy, 0, bvhnodes, aabbs,
            );
            let Some(&(i_tri, _r0, _r1)) = res.first() else {
                continue;
            };
            img[i_h * img_shape.0 + i_w] = i_tri;
        }
    }
    let img = Tensor::from_vec(img, *img_shape, &candle_core::Device::Cpu)?;
    Ok(img)
}
