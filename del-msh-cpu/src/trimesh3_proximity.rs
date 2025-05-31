use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn contacting_pair<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    edge2vtx: &[usize],
    threshold: T,
) -> (Vec<usize>, Vec<T>)
where
    T: Copy + num_traits::Float + 'static + std::fmt::Debug,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let mut contacting_pair = vec![0usize; 0];
    let mut contacting_coord: Vec<T> = vec![];
    // edge-edge
    let num_edge = edge2vtx.len() / 2;
    for i_edge in 0..num_edge {
        for j_edge in i_edge + 1..num_edge {
            let i0 = edge2vtx[i_edge * 2 + 0];
            let i1 = edge2vtx[i_edge * 2 + 1];
            let j0 = edge2vtx[j_edge * 2 + 0];
            let j1 = edge2vtx[j_edge * 2 + 1];
            if i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1 {
                continue;
            };
            use crate::vtx2xyz::to_vec3;
            let a0 = to_vec3(vtx2xyz, i0);
            let a1 = to_vec3(vtx2xyz, i1);
            let b0 = to_vec3(vtx2xyz, j0);
            let b1 = to_vec3(vtx2xyz, j1);
            let (dist, ra1, rb1) = del_geo_core::edge3::nearest_to_edge3(a0, a1, b0, b1);
            if dist > threshold {
                continue;
            }
            let (ra0, rb0) = (T::one() - ra1, T::one() - rb1);
            contacting_pair.extend([i_edge, j_edge, 0]);
            contacting_coord.extend([ra0, ra1, rb0, rb1]);
        }
    }
    // tri-vtx
    let num_tri = tri2vtx.len() / 3;
    let num_vtx = vtx2xyz.len() / 3;
    for i_tri in 0..num_tri {
        for j_vtx in 0..num_vtx {
            let i0 = tri2vtx[i_tri * 3 + 0];
            let i1 = tri2vtx[i_tri * 3 + 1];
            let i2 = tri2vtx[i_tri * 3 + 2];
            if i0 == j_vtx || i1 == j_vtx || i2 == j_vtx {
                continue;
            };
            use crate::vtx2xyz::to_vec3;
            let f0 = to_vec3(vtx2xyz, i0);
            let f1 = to_vec3(vtx2xyz, i1);
            let f2 = to_vec3(vtx2xyz, i2);
            let v0 = to_vec3(vtx2xyz, j_vtx);
            let (_p, rf0, rf1) = del_geo_core::tri3::nearest_to_point3(f0, f1, f2, v0);
            let rf2 = T::one() - rf0 - rf1;
            let p0 = del_geo_core::vec3::add_three(&f0.scale(rf0), &f1.scale(rf1), &f2.scale(rf2));
            let dist = p0.sub(v0).norm();
            if dist > threshold {
                continue;
            }
            contacting_pair.extend([i_tri, j_vtx, 1]);
            contacting_coord.extend([rf0, rf1, rf2, T::one()]);
        }
    }
    (contacting_pair, contacting_coord)
}
