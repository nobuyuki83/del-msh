pub fn normalize2<Real>(
    vtx2xyz: &[nalgebra::Vector2<Real>],
    center_pos: &nalgebra::Vector2<Real>,
    size: Real,
) -> Vec<nalgebra::Vector2<Real>>
where
    Real: nalgebra::RealField + num_traits::Float,
{
    let aabb = del_geo::aabb2::from_vtx2vec(vtx2xyz);
    let cnt = del_geo::aabb2::center(&aabb);
    let max_edge_size = del_geo::aabb2::max_edge_size(&aabb);
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
