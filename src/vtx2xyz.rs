use num_traits::AsPrimitive;

pub fn normalize<Real>(
    vtx2xyz: &[Real]) -> Vec<Real>
    where Real: num_traits::Float + 'static + Copy,
          f64: AsPrimitive<Real>
{
    let aabb = del_geo::aabb3::from_vtx2xyz(vtx2xyz, Real::zero());
    let cnt = del_geo::aabb3::center(&aabb);
    let max_edge_size = del_geo::aabb3::max_edge_size(&aabb);
    let tmp = Real::one() / max_edge_size;
    let mut vtx2xyz_out = Vec::from(vtx2xyz);
    vtx2xyz_out.chunks_mut(3).zip( vtx2xyz.chunks(3) ).for_each(
        |(o,v)| {
            o[0] = (v[0]-cnt[0])*tmp;
            o[1] = (v[1]-cnt[1])*tmp;
            o[2] = (v[2]-cnt[2])*tmp; });
    vtx2xyz_out
}


pub fn from_array_of_nalgebra<T, const N: usize>(
    vecs: &Vec<nalgebra::SVector<T,N>>) -> Vec<T>
where T: nalgebra::RealField + Copy
{
    let mut res = Vec::<T>::with_capacity(vecs.len()*N);
    for vec in vecs { res.extend(vec.iter()); }
    res
}

pub fn cast<T,U>(vtx2xyz0: &[U]) -> Vec<T>
where T: Copy + 'static,
    U: AsPrimitive<T>
{
    let res: Vec<T> = vtx2xyz0.iter().map(|v| v.as_() ).collect();
    res
}

pub fn from_2d_to_3d<Real> (
    vtx2xy: &[Real]) -> Vec<Real>
where Real : num_traits::Zero + Copy
{
    let res : Vec<Real> =  vtx2xy.chunks(2)
        .flat_map(|v| [v[0], v[1], Real::zero()] )
        .collect();
    res
}