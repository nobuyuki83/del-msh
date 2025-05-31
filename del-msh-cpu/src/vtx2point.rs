pub trait HasXyz<Real> {
    fn xyz(&self) -> &[Real; 3];
}

pub fn aabb3_from_points<Real, Point: HasXyz<Real>>(points: &[Point]) -> [Real; 6]
where
    Real: num_traits::Float + std::fmt::Debug,
{
    let mut aabb = [Real::zero(); 6];
    {
        let xyz = points[0].xyz();
        del_geo_core::aabb3::set_as_cube(&mut aabb, xyz, Real::zero());
    }
    for point in points.iter().skip(1) {
        del_geo_core::aabb3::add_point(&mut aabb, point.xyz(), Real::zero());
    }
    assert!(aabb[0] <= aabb[3]);
    assert!(aabb[1] <= aabb[4]);
    assert!(aabb[2] <= aabb[5]);
    aabb
}
