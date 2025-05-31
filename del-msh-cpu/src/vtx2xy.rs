//! functions related to c-style contiguous array of 2D coordinates

use num_traits::AsPrimitive;

pub fn to_vec2<T, Index>(vtx2xyz: &[T], i_vtx: Index) -> &[T; 2]
where
    T: Copy,
    Index: AsPrimitive<usize>,
{
    let i_vtx: usize = i_vtx.as_();
    arrayref::array_ref![vtx2xyz, i_vtx * 2, 2]
}

pub fn to_vtx2xyz<Real>(vtx2xy: &[Real]) -> Vec<Real>
where
    Real: num_traits::Zero + Copy,
{
    let res: Vec<Real> = vtx2xy
        .chunks(2)
        .flat_map(|v| [v[0], v[1], Real::zero()])
        .collect();
    res
}

pub fn aabb2<Real>(vtx2xy: &[Real]) -> [Real; 4]
where
    Real: num_traits::Float,
{
    let mut aabb = [vtx2xy[0], vtx2xy[1], vtx2xy[0], vtx2xy[1]];
    vtx2xy.chunks(2).skip(1).for_each(|v| {
        aabb[0] = if v[0] < aabb[0] { v[0] } else { aabb[0] };
        aabb[1] = if v[1] < aabb[1] { v[1] } else { aabb[1] };
        aabb[2] = if v[0] > aabb[2] { v[0] } else { aabb[2] };
        aabb[3] = if v[1] > aabb[3] { v[1] } else { aabb[3] };
    });
    aabb
}

pub fn aabb2_indexed<Index, T>(idx2vtx: &[Index], vtx2xy: &[T], eps: T) -> [T; 4]
where
    T: num_traits::Float,
    Index: AsPrimitive<usize>,
{
    assert!(!idx2vtx.is_empty());
    let mut aabb = [T::zero(); 4];
    {
        let i_vtx: usize = idx2vtx[0].as_();
        {
            let cgx = vtx2xy[i_vtx * 2];
            aabb[0] = cgx - eps;
            aabb[2] = cgx + eps;
        }
        {
            let cgy = vtx2xy[i_vtx * 2 + 1];
            aabb[1] = cgy - eps;
            aabb[3] = cgy + eps;
        }
    }
    for &i_vtx in idx2vtx.iter().skip(1) {
        let i_vtx = i_vtx.as_();
        {
            let cgx = vtx2xy[i_vtx * 2];
            aabb[0] = if cgx - eps < aabb[0] {
                cgx - eps
            } else {
                aabb[0]
            };
            aabb[2] = if cgx + eps > aabb[2] {
                cgx + eps
            } else {
                aabb[2]
            };
        }
        {
            let cgy = vtx2xy[i_vtx * 2 + 1];
            aabb[1] = if cgy - eps < aabb[1] {
                cgy - eps
            } else {
                aabb[1]
            };
            aabb[3] = if cgy + eps > aabb[3] {
                cgy + eps
            } else {
                aabb[3]
            };
        }
    }
    assert!(aabb[0] <= aabb[2]);
    assert!(aabb[1] <= aabb[3]);
    aabb
}

pub fn normalize<Real>(vtx2xy: &[Real], center_pos: &[Real; 2], size: Real) -> Vec<Real>
where
    Real: num_traits::Float,
{
    let aabb = aabb2(vtx2xy);
    let cnt = del_geo_core::aabb2::center(&aabb);
    let max_edge_size = del_geo_core::aabb2::max_edge_size(&aabb);
    let tmp = size / max_edge_size;
    vtx2xy
        .chunks(2)
        .flat_map(|v| {
            [
                (v[0] - cnt[0]) * tmp + center_pos[0],
                (v[1] - cnt[1]) * tmp + center_pos[1],
            ]
        })
        .collect()
    /*
    let mut vtx2xyz_out = Vec::from(vtx2xy);
    vtx2xyz_out
        .iter_mut()
        .zip(vtx2xy.iter())
        .for_each(|(o, v)| {

        });
    vtx2xyz_out
         */
}
