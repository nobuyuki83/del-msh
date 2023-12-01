pub struct IntersectingPair {
    pub i_tri: usize,
    pub j_tri: usize,
    pub p0: nalgebra::Vector3::<f32>,
    pub p1: nalgebra::Vector3::<f32>,
}

#[allow(clippy::identity_op)]
pub fn intersection_triangle_mesh_between_bvh_branches(
    pairs: &mut Vec<IntersectingPair>,
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    ibvh0: usize,
    ibvh1: usize,
    bvhnodes: &[usize],
    aabbs: &[f32])
{
    assert!(ibvh0 < aabbs.len() / 6);
    assert!(ibvh1 < aabbs.len() / 6);
    if !del_geo::aabb3::is_intersect(
        &aabbs[ibvh0 * 6..(ibvh0 + 1) * 6],
        &aabbs[ibvh1 * 6..(ibvh1 + 1) * 6]) {
        return;
    }
    let ichild0_0 = bvhnodes[ibvh0 * 3 + 1];
    let ichild0_1 = bvhnodes[ibvh0 * 3 + 2];
    let ichild1_0 = bvhnodes[ibvh1 * 3 + 1];
    let ichild1_1 = bvhnodes[ibvh1 * 3 + 2];
    let is_leaf0 = ichild0_1 == usize::MAX;
    let is_leaf1 = ichild1_1 == usize::MAX;
    if !is_leaf0 && !is_leaf1 {
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_0, ichild1_0, bvhnodes, aabbs);
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_1, ichild1_0, bvhnodes, aabbs);
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_0, ichild1_1, bvhnodes, aabbs);
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_1, ichild1_1, bvhnodes, aabbs);
    } else if !is_leaf0 && is_leaf1 {
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_0, ibvh1, bvhnodes, aabbs);
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_1, ibvh1, bvhnodes, aabbs);
    } else if is_leaf0 && !is_leaf1 {
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ibvh0, ichild1_0, bvhnodes, aabbs);
        intersection_triangle_mesh_between_bvh_branches(
            pairs, tri2vtx, vtx2xyz, ibvh0, ichild1_1, bvhnodes, aabbs);
    } else if is_leaf0 && is_leaf1 {
        let itri = ichild0_0;
        let jtri = ichild1_0;
        let i0 = tri2vtx[itri * 3 + 0];
        let i1 = tri2vtx[itri * 3 + 1];
        let i2 = tri2vtx[itri * 3 + 2];
        let j0 = tri2vtx[jtri * 3 + 0];
        let j1 = tri2vtx[jtri * 3 + 1];
        let j2 = tri2vtx[jtri * 3 + 2];
        if i0 == j0 || i0 == j1 || i0 == j2 { return; };
        if i1 == j0 || i1 == j1 || i1 == j2 { return; };
        if i2 == j0 || i2 == j1 || i2 == j2 { return; };
        let res = del_geo::tri3::is_intersection_tri3(
            &nalgebra::Vector3::<f32>::from_row_slice(&vtx2xyz[i0 * 3..(i0 + 1) * 3]),
            &nalgebra::Vector3::<f32>::from_row_slice(&vtx2xyz[i1 * 3..(i1 + 1) * 3]),
            &nalgebra::Vector3::<f32>::from_row_slice(&vtx2xyz[i2 * 3..(i2 + 1) * 3]),
            &nalgebra::Vector3::<f32>::from_row_slice(&vtx2xyz[j0 * 3..(j0 + 1) * 3]),
            &nalgebra::Vector3::<f32>::from_row_slice(&vtx2xyz[j1 * 3..(j1 + 1) * 3]),
            &nalgebra::Vector3::<f32>::from_row_slice(&vtx2xyz[j2 * 3..(j2 + 1) * 3]));
        if let Some((p0, p1)) = res {
            let itp = IntersectingPair {
                i_tri: itri,
                j_tri: jtri,
                p0,
                p1,
            };
            pairs.push(itp);
        } else { return; }
    }
}

pub fn intersection_triangle_mesh_inside_branch(
    tripairs: &mut Vec<IntersectingPair>,
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    ibvh: usize,
    bvhnodes: &[usize],
    aabbs: &[f32])
{
    let ichild0 = bvhnodes[ibvh * 3 + 1];
    let ichild1 = bvhnodes[ibvh * 3 + 2];
    if ichild1 == usize::MAX { return; }
    intersection_triangle_mesh_between_bvh_branches(
        tripairs, tri2vtx, vtx2xyz, ichild0, ichild1, bvhnodes, aabbs);
    intersection_triangle_mesh_inside_branch(
        tripairs, tri2vtx, vtx2xyz, ichild0, bvhnodes, aabbs);
    intersection_triangle_mesh_inside_branch(
        tripairs, tri2vtx, vtx2xyz, ichild1, bvhnodes, aabbs);
}


#[cfg(test)]
mod tests {
    use crate::{elem2center, elem2elem};
    use crate::bvh3_intersection_self::IntersectingPair;

    #[test]
    fn test0() {
        let (tri2vtx, vtx2xyz)
            = crate::trimesh3_primitive::from_sphere(1.0, 16, 32);
        let (face2idx, idx2node) = elem2elem::face2node_of_polygon_element(3);
        let tri2tri = elem2elem::from_uniform_mesh(
            &tri2vtx, 3, &face2idx, &idx2node, vtx2xyz.len() / 3);
        let tri2center = elem2center::from_uniform_mesh(
            &tri2vtx, 3, &vtx2xyz, 3);
        let bvhnodes = crate::bvh3::build_topology_for_uniform_mesh_with_elem2elem_elem2center(
            &tri2tri, 3, &tri2center);
        let mut aabb = Vec::<f32>::new();
        aabb.resize(bvhnodes.len() / 3 * 6, 0.);
        crate::bvh3::build_geometry_aabb_for_uniform_mesh(
            &mut aabb, 0, &bvhnodes, &tri2vtx, 3, &vtx2xyz);
        let mut pairs = Vec::<IntersectingPair>::new();
        crate::bvh3_intersection_self::intersection_triangle_mesh_inside_branch(
            &mut pairs, &tri2vtx, &vtx2xyz, 0, &bvhnodes, &aabb);
        assert_eq!(pairs.len(), 0);
    }
}