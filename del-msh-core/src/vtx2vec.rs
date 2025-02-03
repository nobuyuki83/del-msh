pub fn normalize2<Real>(vtx2xyz: &[[Real; 2]], center_pos: &[Real; 2], size: Real) -> Vec<[Real; 2]>
where
    Real: num_traits::Float,
{
    let aabb = crate::vtx2vec::aabb2(vtx2xyz);
    let cnt = del_geo_core::aabb2::center(&aabb);
    let max_edge_size = del_geo_core::aabb2::max_edge_size(&aabb);
    let tmp = size / max_edge_size;
    let mut vtx2xyz_out = Vec::from(vtx2xyz);
    vtx2xyz_out
        .iter_mut()
        .zip(vtx2xyz.iter())
        .for_each(|(o, v)| {
            o[0] = (v[0] - cnt[0]) * tmp + center_pos[0];
            o[1] = (v[1] - cnt[1]) * tmp + center_pos[1];
        });
    vtx2xyz_out
}

pub fn aabb2<T>(vtx2vec: &[[T; 2]]) -> [T; 4]
where
    T: num_traits::Float + Copy,
{
    let mut aabb = [vtx2vec[0][0], vtx2vec[0][1], vtx2vec[0][0], vtx2vec[0][1]];
    for xy in vtx2vec.iter().skip(1) {
        aabb[0] = aabb[0].min(xy[0]);
        aabb[1] = aabb[1].min(xy[1]);
        aabb[2] = aabb[2].max(xy[0]);
        aabb[3] = aabb[3].max(xy[1]);
    }
    aabb
}
