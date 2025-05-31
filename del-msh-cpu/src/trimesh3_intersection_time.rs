use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn edge_edge_between_bvh_branches<T>(
    edge2vtx: &[usize],
    vtx2xyz0: &[T],
    vtx2xyz1: &[T],
    ibvh0: usize,
    ibvh1: usize,
    bvhnodes: &[usize],
    aabbs: &[T],
) where
    T: Copy + num_traits::Float + 'static + std::fmt::Debug + std::fmt::Display,
    i64: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    assert!(ibvh0 < aabbs.len() / 6);
    assert!(ibvh1 < aabbs.len() / 6);
    // trim branch
    if !del_geo_core::aabb3::is_intersect(
        arrayref::array_ref![aabbs, ibvh0 * 6, 6],
        arrayref::array_ref![aabbs, ibvh1 * 6, 6],
    ) {
        return;
    }
    let ichild0_left = bvhnodes[ibvh0 * 3 + 1];
    let ichild0_right = bvhnodes[ibvh0 * 3 + 2];
    let ichild1_left = bvhnodes[ibvh1 * 3 + 1];
    let ichild1_right = bvhnodes[ibvh1 * 3 + 2];
    let is_leaf0 = ichild0_right == usize::MAX;
    let is_leaf1 = ichild1_right == usize::MAX;
    if !is_leaf0 && !is_leaf1 {
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ichild0_left,
            ichild1_left,
            bvhnodes,
            aabbs,
        );
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ichild0_right,
            ichild1_left,
            bvhnodes,
            aabbs,
        );
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ichild0_left,
            ichild1_right,
            bvhnodes,
            aabbs,
        );
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ichild0_right,
            ichild1_right,
            bvhnodes,
            aabbs,
        );
    } else if !is_leaf0 && is_leaf1 {
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ichild0_left,
            ibvh1,
            bvhnodes,
            aabbs,
        );
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ichild0_right,
            ibvh1,
            bvhnodes,
            aabbs,
        );
    } else if is_leaf0 && !is_leaf1 {
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ibvh0,
            ichild1_left,
            bvhnodes,
            aabbs,
        );
        edge_edge_between_bvh_branches(
            edge2vtx,
            vtx2xyz0,
            vtx2xyz1,
            ibvh0,
            ichild1_right,
            bvhnodes,
            aabbs,
        );
    } else if is_leaf0 && is_leaf1 {
        // check the primitive ccd
        let i_edge = ichild0_left;
        let j_edge = ichild1_left;
        let i0 = edge2vtx[i_edge * 2 + 0];
        let i1 = edge2vtx[i_edge * 2 + 1];
        let j0 = edge2vtx[j_edge * 2 + 0];
        let j1 = edge2vtx[j_edge * 2 + 1];
        if i0 == j0 || i0 == j1 {
            return;
        };
        if i1 == j0 || i1 == j1 {
            return;
        };
        use crate::vtx2xyz::to_vec3;
        let a0s = to_vec3(vtx2xyz0, i0);
        let a1s = to_vec3(vtx2xyz0, i1);
        let b0s = to_vec3(vtx2xyz0, j0);
        let b1s = to_vec3(vtx2xyz0, j1);
        let a0e = to_vec3(vtx2xyz1, i0);
        let a1e = to_vec3(vtx2xyz1, i1);
        let b0e = to_vec3(vtx2xyz1, j0);
        let b1e = to_vec3(vtx2xyz1, j1);
        let t = del_geo_core::ccd3::intersecting_time_ee(
            del_geo_core::ccd3::EdgeEdge {
                a0: a0s,
                a1: a1s,
                b0: b0s,
                b1: b1s,
            },
            del_geo_core::ccd3::EdgeEdge {
                a0: a0e,
                a1: a1e,
                b0: b0e,
                b1: b1e,
            },
            1.0e-5f64.as_(),
        );
        if let Some(t) = t {
            dbg!(t);
        }
    }
}

pub fn edge_edge_inside_branch<T>(
    edge2vtx: &[usize],
    vtx2xyz0: &[T],
    vtx2xyz1: &[T],
    ibvh: usize,
    bvhnodes: &[usize],
    aabbs: &[T],
) where
    T: Copy + num_traits::Float + 'static + std::fmt::Display + std::fmt::Debug,
    i64: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let ichild_left = bvhnodes[ibvh * 3 + 1];
    let ichild_right = bvhnodes[ibvh * 3 + 2];
    if ichild_right == usize::MAX {
        return;
    } // ibvh is a leaf node
    edge_edge_between_bvh_branches(
        edge2vtx,
        vtx2xyz0,
        vtx2xyz1,
        ichild_left,
        ichild_right,
        bvhnodes,
        aabbs,
    );
    edge_edge_inside_branch(edge2vtx, vtx2xyz0, vtx2xyz1, ichild_left, bvhnodes, aabbs);
    edge_edge_inside_branch(edge2vtx, vtx2xyz0, vtx2xyz1, ichild_right, bvhnodes, aabbs);
}

pub fn search_with_bvh<T>(
    edge2vtx: &[usize],
    tri2vtx: &[usize],
    vtx2xyz0: &[T],
    vtx2xyz1: &[T],
    _bvhnodes: &[usize],
    _aabbs: &[T],
) -> (Vec<usize>, Vec<T>)
where
    T: num_traits::Float + 'static + std::fmt::Debug + std::fmt::Display,
    i64: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    /*
    let mut intersection_pair = vec!(0usize; 0);
    let mut intersection_times = vec!(T::zero(); 0);
    assert_eq!(vtx2xyz0.len(), vtx2xyz1.len());
    intersection_pair.clear();
    intersection_times.clear();
    (intersection_pair, intersection_times)
     */
    // TODO: implement using BVH
    search_brute_force(edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1, 1.0e-5.as_())
}

#[allow(clippy::identity_op)]
pub fn search_brute_force<T>(
    edge2vtx: &[usize],
    tri2vtx: &[usize],
    vtx2xyz0: &[T],
    vtx2xyz1: &[T],
    epsilon: T,
) -> (Vec<usize>, Vec<T>)
where
    T: num_traits::Float + 'static + std::fmt::Display + std::fmt::Debug,
    i64: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let mut intersection_pair = vec![0usize; 0];
    let mut intersection_times: Vec<T> = vec![];
    assert_eq!(vtx2xyz0.len(), vtx2xyz1.len());
    intersection_pair.clear();
    intersection_times.clear();
    // edge_edge_inside_branch(edge2vtx, vtx2xyz0, vtx2xyz1, roots[1], bvhnodes, aabbs);
    //
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
            let a0s = to_vec3(vtx2xyz0, i0);
            let a1s = to_vec3(vtx2xyz0, i1);
            let b0s = to_vec3(vtx2xyz0, j0);
            let b1s = to_vec3(vtx2xyz0, j1);
            let a0e = to_vec3(vtx2xyz1, i0);
            let a1e = to_vec3(vtx2xyz1, i1);
            let b0e = to_vec3(vtx2xyz1, j0);
            let b1e = to_vec3(vtx2xyz1, j1);
            let t = del_geo_core::ccd3::intersecting_time_ee(
                del_geo_core::ccd3::EdgeEdge {
                    a0: a0s,
                    a1: a1s,
                    b0: b0s,
                    b1: b1s,
                },
                del_geo_core::ccd3::EdgeEdge {
                    a0: a0e,
                    a1: a1e,
                    b0: b0e,
                    b1: b1e,
                },
                epsilon,
            );
            if let Some(t) = t {
                // println!("ee {} {} {}", t, i_edge, j_edge);
                intersection_pair.extend([i_edge, j_edge, 0]);
                intersection_times.push(t);
            }
        }
    }
    //
    let num_tri = tri2vtx.len() / 3;
    let num_vtx = vtx2xyz0.len() / 3;
    for i_tri in 0..num_tri {
        for j_vtx in 0..num_vtx {
            let i0 = tri2vtx[i_tri * 3 + 0];
            let i1 = tri2vtx[i_tri * 3 + 1];
            let i2 = tri2vtx[i_tri * 3 + 2];
            if i0 == j_vtx || i1 == j_vtx || i2 == j_vtx {
                continue;
            };
            use crate::vtx2xyz::to_vec3;
            let f0s = to_vec3(vtx2xyz0, i0);
            let f1s = to_vec3(vtx2xyz0, i1);
            let f2s = to_vec3(vtx2xyz0, i2);
            let v0s = to_vec3(vtx2xyz0, j_vtx);
            let f0e = to_vec3(vtx2xyz1, i0);
            let f1e = to_vec3(vtx2xyz1, i1);
            let f2e = to_vec3(vtx2xyz1, i2);
            let v0e = to_vec3(vtx2xyz1, j_vtx);
            let t = del_geo_core::ccd3::intersecting_time_fv(
                del_geo_core::ccd3::FaceVertex {
                    f0: f0s,
                    f1: f1s,
                    f2: f2s,
                    v: v0s,
                },
                del_geo_core::ccd3::FaceVertex {
                    f0: f0e,
                    f1: f1e,
                    f2: f2e,
                    v: v0e,
                },
                epsilon,
            );
            if let Some(t) = t {
                // println!("fv {} {} {}", t, i_tri, j_vtx);
                intersection_pair.extend([i_tri, j_vtx, 1]);
                intersection_times.push(t);
            }
        }
    }
    (intersection_pair, intersection_times)
}

#[allow(clippy::identity_op)]
pub fn print_intersection<T>(intersection_pair: &[usize], intersection_time: &[T])
where
    T: std::fmt::Display + Copy,
{
    for i in 0..intersection_pair.len() / 3 {
        if intersection_pair[i * 3 + 2] == 0 {
            // edge-edge
            println!(
                "edge-edge {} {} ## {}",
                intersection_pair[i * 3 + 0],
                intersection_pair[i * 3 + 1],
                intersection_time[i]
            );
        } else {
            println!(
                "face-vtx {} {} ## {}",
                intersection_pair[i * 3 + 0],
                intersection_pair[i * 3 + 1],
                intersection_time[i]
            );
        }
    }
}
