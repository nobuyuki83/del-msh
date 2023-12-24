use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn intersection_time<T>(
    intersection_pair: &mut Vec<usize>,
    intersection_times: &mut Vec<T>,
    edge2vtx: &[usize],
    tri2vtx: &[usize],
    vtx2xyz0: &[T],
    vtx2xyz1: &[T],
    bvhnodes: &[usize],
    aabbs: &[T],
    roots: &[usize])
    where T: num_traits::Float + nalgebra::RealField,
          i64: AsPrimitive<T>,
          f64: AsPrimitive<T>
{
    assert_eq!(vtx2xyz0.len(), vtx2xyz1.len());
    assert!(roots.len() >= 3);
    assert!(roots[0] < bvhnodes.len() / 3);
    assert!(roots[1] < bvhnodes.len() / 3);
    assert!(roots[2] < bvhnodes.len() / 3);
    assert_eq!(aabbs.len() / 6, bvhnodes.len() / 3);
    intersection_pair.clear();
    intersection_times.clear();
    // edge_edge_inside_branch(edge2vtx, vtx2xyz0, vtx2xyz1, roots[1], bvhnodes, aabbs);
    //
    let num_edge = edge2vtx.len() / 2;
    for i_edge in 0..num_edge {
        for j_edge in i_edge+1..num_edge {
            let i0 = edge2vtx[i_edge * 2 + 0];
            let i1 = edge2vtx[i_edge * 2 + 1];
            let j0 = edge2vtx[j_edge * 2 + 0];
            let j1 = edge2vtx[j_edge * 2 + 1];
            if i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1 { continue; };
            use del_geo::vec3::navec3;
            let a0s = navec3(vtx2xyz0, i0);
            let a1s = navec3(vtx2xyz0, i1);
            let b0s = navec3(vtx2xyz0, j0);
            let b1s = navec3(vtx2xyz0, j1);
            let a0e = navec3(vtx2xyz1, i0);
            let a1e = navec3(vtx2xyz1, i1);
            let b0e = navec3(vtx2xyz1, j0);
            let b1e = navec3(vtx2xyz1, j1);
            let t = del_geo::ccd::intersecting_time_ee(
                &a0s, &a1s, &b0s, &b1s,
                &a0e, &a1e, &b0e, &b1e,
                1.0e-5f64.as_());
            if let Some(t) = t {
                intersection_pair.push(i_edge);
                intersection_pair.push(j_edge);
                intersection_pair.push(0);
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
            if i0 == j_vtx || i1 == j_vtx || i2 == j_vtx { continue; };
            use del_geo::vec3::navec3;
            let f0s = navec3(vtx2xyz0, i0);
            let f1s = navec3(vtx2xyz0, i1);
            let f2s = navec3(vtx2xyz0, i2);
            let v0s = navec3(vtx2xyz0, j_vtx);
            let f0e = navec3(vtx2xyz1, i0);
            let f1e = navec3(vtx2xyz1, i1);
            let f2e = navec3(vtx2xyz1, i2);
            let v0e = navec3(vtx2xyz1, j_vtx);
            let t = del_geo::ccd::intersecting_time_fv(
                &f0s, &f1s, &f2s, &v0s,
                &f0e, &f1e, &f2e, &v0e,
                1.0e-5f64.as_());
            if let Some(t) = t {
                intersection_pair.push(i_tri);
                intersection_pair.push(j_vtx);
                intersection_pair.push(1);
                intersection_times.push(t);
            }
        }
    }
}

#[allow(clippy::identity_op)]
pub fn edge_edge_between_bvh_branches<T>(
    edge2vtx: &[usize],
    vtx2xyz0: &[T],
    vtx2xyz1: &[T],
    ibvh0: usize,
    ibvh1: usize,
    bvhnodes: &[usize],
    aabbs: &[T])
    where T: nalgebra::RealField + Copy + num_traits::Float,
          i64: AsPrimitive<T>,
          f64: AsPrimitive<T>
{
    assert!(ibvh0 < aabbs.len() / 6);
    assert!(ibvh1 < aabbs.len() / 6);
    if !del_geo::aabb3::is_intersect( // trim branch
        &aabbs[ibvh0 * 6..(ibvh0 + 1) * 6],
        &aabbs[ibvh1 * 6..(ibvh1 + 1) * 6]) {
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
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ichild0_left, ichild1_left, bvhnodes, aabbs);
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ichild0_right, ichild1_left, bvhnodes, aabbs);
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ichild0_left, ichild1_right, bvhnodes, aabbs);
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ichild0_right, ichild1_right, bvhnodes, aabbs);
    } else if !is_leaf0 && is_leaf1 {
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ichild0_left, ibvh1, bvhnodes, aabbs);
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ichild0_right, ibvh1, bvhnodes, aabbs);
    } else if is_leaf0 && !is_leaf1 {
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ibvh0, ichild1_left, bvhnodes, aabbs);
        edge_edge_between_bvh_branches(
            edge2vtx, vtx2xyz0, vtx2xyz1,
            ibvh0, ichild1_right, bvhnodes, aabbs);
    } else if is_leaf0 && is_leaf1 { // check the primitive ccd
        let i_edge = ichild0_left;
        let j_edge = ichild1_left;
        let i0 = edge2vtx[i_edge * 2 + 0];
        let i1 = edge2vtx[i_edge * 2 + 1];
        let j0 = edge2vtx[j_edge * 2 + 0];
        let j1 = edge2vtx[j_edge * 2 + 1];
        if i0 == j0 || i0 == j1 { return; };
        if i1 == j0 || i1 == j1 { return; };
        use del_geo::vec3::navec3;
        let a0s = navec3(vtx2xyz0, i0);
        let a1s = navec3(vtx2xyz0, i1);
        let b0s = navec3(vtx2xyz0, j0);
        let b1s = navec3(vtx2xyz0, j1);
        let a0e = navec3(vtx2xyz1, i0);
        let a1e = navec3(vtx2xyz1, i1);
        let b0e = navec3(vtx2xyz1, j0);
        let b1e = navec3(vtx2xyz1, j1);
        let t = del_geo::ccd::intersecting_time_ee(
            &a0s, &a1s, &b0s, &b1s,
            &a0e, &a1e, &b0e, &b1e,
            1.0e-5f64.as_());
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
    aabbs: &[T])
    where T: nalgebra::RealField + Copy + num_traits::Float,
          i64: AsPrimitive<T>,
          f64: AsPrimitive<T>
{
    let ichild_left = bvhnodes[ibvh * 3 + 1];
    let ichild_right = bvhnodes[ibvh * 3 + 2];
    if ichild_right == usize::MAX { return; } // ibvh is a leaf node
    edge_edge_between_bvh_branches(
        edge2vtx, vtx2xyz0, vtx2xyz1, ichild_left, ichild_right, bvhnodes, aabbs);
    edge_edge_inside_branch(
        edge2vtx, vtx2xyz0, vtx2xyz1, ichild_left, bvhnodes, aabbs);
    edge_edge_inside_branch(
        edge2vtx, vtx2xyz0, vtx2xyz1, ichild_right, bvhnodes, aabbs);
}