use num_traits::AsPrimitive;

pub fn from_array_of_nalgebra<T, const N: usize>(vtx2vecn: &Vec<nalgebra::SVector<T, N>>) -> Vec<T>
where
    T: nalgebra::RealField + Copy,
{
    let mut res = Vec::<T>::with_capacity(vtx2vecn.len() * N);
    for vec in vtx2vecn {
        res.extend(vec.iter());
    }
    res
}

pub fn to_array_of_nalgebra_vector<T, const N: usize>(vtx2xyz: &[T]) -> Vec<nalgebra::SVector<T, N>>
where
    T: nalgebra::RealField,
{
    vtx2xyz
        .chunks(N)
        .map(|v| nalgebra::SVector::<T, N>::from_row_slice(v))
        .collect()
}

pub fn cast<T, U>(vtx2xyz0: &[U]) -> Vec<T>
where
    T: Copy + 'static,
    U: AsPrimitive<T>,
{
    let res: Vec<T> = vtx2xyz0.iter().map(|v| v.as_()).collect();
    res
}

pub fn from_2d_to_3d<Real>(vtx2xy: &[Real]) -> Vec<Real>
where
    Real: num_traits::Zero + Copy,
{
    let res: Vec<Real> = vtx2xy
        .chunks(2)
        .flat_map(|v| [v[0], v[1], Real::zero()])
        .collect();
    res
}

pub fn translate_and_scale<Real>(
    vtx2xyz_out: &mut [Real],
    vtx2xyz_in: &[Real],
    transl: &[Real; 3],
    scale: Real,
) where
    Real: num_traits::Float,
{
    assert_eq!(vtx2xyz_in.len(), vtx2xyz_out.len());
    vtx2xyz_out
        .chunks_mut(3)
        .zip(vtx2xyz_in.chunks(3))
        .for_each(|(o, v)| {
            o[0] = (v[0] + transl[0]) * scale;
            o[1] = (v[1] + transl[1]) * scale;
            o[2] = (v[2] + transl[2]) * scale;
        });
    /*
    let num_vtx = vtx2xyz_in.len() / 3;
    for i_vtx in 0..num_vtx {
        let p = &vtx2xyz_in[i_vtx * 3..(i_vtx + 1) * 3];
        let q = &mut vtx2xyz_out[i_vtx * 3..(i_vtx + 1) * 3];
        q[0] = (p[0] + transl[0]) * scale;
        q[1] = (p[1] + transl[1]) * scale;
        q[2] = (p[2] + transl[2]) * scale;
    }
     */
}

pub fn normalize<Real>(vtx2xyz: &[Real]) -> Vec<Real>
where
    Real: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<Real>,
{
    let aabb = del_geo::aabb3::from_vtx2xyz(vtx2xyz, Real::zero());
    let cnt = del_geo::aabb3::center(&aabb);
    let transl = [-cnt[0], -cnt[1], -cnt[2]];
    let max_edge_size = del_geo::aabb3::max_edge_size(&aabb);
    let scale = Real::one() / max_edge_size;
    let mut vtx2xyz_out = Vec::from(vtx2xyz);
    translate_and_scale(&mut vtx2xyz_out, vtx2xyz, &transl, scale);
    vtx2xyz_out
}
