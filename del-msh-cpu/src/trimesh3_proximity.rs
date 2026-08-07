use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn contacting_pair<T>(
    tri2vtx: &[[usize; 3]],
    vtx2xyz: &[[T; 3]],
    edge2vtx: &[[usize; 2]],
    threshold: T,
) -> (Vec<[usize; 3]>, Vec<[T; 4]>)
where
    T: Copy + num_traits::Float + 'static + std::fmt::Debug,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let mut contacting_pair = vec![[0usize; 3]; 0];
    let mut contacting_coord: Vec<[T; 4]> = vec![];
    // edge-edge
    let num_edge = edge2vtx.len();
    for i_edge in 0..num_edge {
        for j_edge in i_edge + 1..num_edge {
            let i0 = edge2vtx[i_edge][0];
            let i1 = edge2vtx[i_edge][1];
            let j0 = edge2vtx[j_edge][0];
            let j1 = edge2vtx[j_edge][1];
            if i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1 {
                continue;
            };
            let a0 = &vtx2xyz[i0];
            let a1 = &vtx2xyz[i1];
            let b0 = &vtx2xyz[j0];
            let b1 = &vtx2xyz[j1];
            let (dist, ra1, rb1) = del_geo_core::edge3::nearest_to_edge3(a0, a1, b0, b1);
            if dist > threshold {
                continue;
            }
            let (ra0, rb0) = (T::one() - ra1, T::one() - rb1);
            contacting_pair.push([i_edge, j_edge, 0]);
            contacting_coord.push([ra0, ra1, rb0, rb1]);
        }
    }
    // tri-vtx
    let num_vtx = vtx2xyz.len();
    for (i_tri, node2vtx) in tri2vtx.iter().enumerate() {
        for j_vtx in 0..num_vtx {
            let i0 = node2vtx[0];
            let i1 = node2vtx[1];
            let i2 = node2vtx[2];
            if i0 == j_vtx || i1 == j_vtx || i2 == j_vtx {
                continue;
            };
            let f0 = &vtx2xyz[i0];
            let f1 = &vtx2xyz[i1];
            let f2 = &vtx2xyz[i2];
            let v0 = &vtx2xyz[j_vtx];
            let (_p, rf0, rf1) = del_geo_core::tri3::nearest_to_point3(f0, f1, f2, v0);
            let rf2 = T::one() - rf0 - rf1;
            let p0 = del_geo_core::vec3::add_three(&f0.scale(rf0), &f1.scale(rf1), &f2.scale(rf2));
            let dist = p0.sub(v0).norm();
            if dist > threshold {
                continue;
            }
            contacting_pair.push([i_tri, j_vtx, 1]);
            contacting_coord.push([rf0, rf1, rf2, T::one()]);
        }
    }
    (contacting_pair, contacting_coord)
}
