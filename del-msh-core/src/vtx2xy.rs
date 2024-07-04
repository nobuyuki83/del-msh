//! functions for contiguous c-style array of 2D coordinates

use num_traits::AsPrimitive;

pub fn to_array2<T, Index>(vtx2xyz: &[T], i_vtx: Index) -> [T; 2]
where
    T: Copy,
    Index: AsPrimitive<usize>,
{
    let i_vtx: usize = i_vtx.as_();
    [vtx2xyz[i_vtx * 2], vtx2xyz[i_vtx * 2 + 1]]
}

pub fn to_navec2<T>(vtx2xyz: &[T], i_vtx: usize) -> nalgebra::Vector2<T>
where
    T: Copy + nalgebra::RealField,
{
    nalgebra::Vector2::<T>::from_row_slice(&vtx2xyz[i_vtx * 2..(i_vtx + 1) * 2])
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
