//! method for finding self-intersection using BVH

pub struct IntersectingPair<T> {
    pub i_tri: usize,
    pub j_tri: usize,
    pub p0: nalgebra::Vector3<T>,
    pub p1: nalgebra::Vector3<T>,
}

#[allow(clippy::identity_op)]
fn intersection_of_two_triangles_in_mesh<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    i_tri: usize,
    j_tri: usize,
) -> Option<(nalgebra::Vector3<T>, nalgebra::Vector3<T>)>
where
    T: Copy + nalgebra::RealField,
{
    let i0 = tri2vtx[i_tri * 3 + 0];
    let i1 = tri2vtx[i_tri * 3 + 1];
    let i2 = tri2vtx[i_tri * 3 + 2];
    let j0 = tri2vtx[j_tri * 3 + 0];
    let j1 = tri2vtx[j_tri * 3 + 1];
    let j2 = tri2vtx[j_tri * 3 + 2];
    let icnt0 = if i0 == j0 || i0 == j1 || i0 == j2 {
        1
    } else {
        0
    };
    let icnt1 = if i1 == j0 || i1 == j1 || i1 == j2 {
        1
    } else {
        0
    };
    let icnt2 = if i2 == j0 || i2 == j1 || i2 == j2 {
        1
    } else {
        0
    };
    if icnt0 + icnt1 + icnt2 > 1 {
        return None;
    } // return  if sharing edge, identical triangle
    use crate::vtx2xyz::to_navec3;
    if icnt0 + icnt1 + icnt2 == 0 {
        del_geo_nalgebra::tri3::is_intersection_tri3(
            &to_navec3(vtx2xyz, i0),
            &to_navec3(vtx2xyz, i1),
            &to_navec3(vtx2xyz, i2),
            &to_navec3(vtx2xyz, j0),
            &to_navec3(vtx2xyz, j1),
            &to_navec3(vtx2xyz, j2),
        )
    } else {
        // sharing one point
        // compute permutation
        let is = if icnt0 == 1 {
            0
        } else if icnt1 == 1 {
            1
        } else {
            2
        };
        let jcnt0 = if j0 == i0 || j0 == i1 || j0 == i2 {
            1
        } else {
            0
        };
        let jcnt1 = if j1 == i0 || j1 == i1 || j1 == i2 {
            1
        } else {
            0
        };
        let jcnt2 = if j2 == i0 || j2 == i1 || j2 == i2 {
            1
        } else {
            0
        };
        assert_eq!(jcnt0 + jcnt1 + jcnt2, 1);
        let js = if jcnt0 == 1 {
            0
        } else if jcnt1 == 1 {
            1
        } else {
            2
        };
        let node2vtx_i = [i0, i1, i2];
        let node2vtx_j = [j0, j1, j2];
        assert_eq!(node2vtx_i[is], node2vtx_j[js]);
        del_geo_nalgebra::tri3::is_intersection_tri3(
            &to_navec3(vtx2xyz, node2vtx_i[(is + 0) % 3]),
            &to_navec3(vtx2xyz, node2vtx_i[(is + 1) % 3]),
            &to_navec3(vtx2xyz, node2vtx_i[(is + 2) % 3]),
            &to_navec3(vtx2xyz, node2vtx_j[(js + 0) % 3]),
            &to_navec3(vtx2xyz, node2vtx_j[(js + 1) % 3]),
            &to_navec3(vtx2xyz, node2vtx_j[(js + 2) % 3]),
        )
    }
}

#[allow(clippy::identity_op)]
pub fn search_with_bvh_between_branches<T>(
    pairs: &mut Vec<IntersectingPair<T>>,
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    ibvh0: usize,
    ibvh1: usize,
    bvhnodes: &[usize],
    aabbs: &[T],
) where
    T: nalgebra::RealField + Copy,
{
    assert!(ibvh0 < aabbs.len() / 6);
    assert!(ibvh1 < aabbs.len() / 6);
    if !del_geo_core::aabb3::is_intersect(
        (&aabbs[ibvh0 * 6..(ibvh0 + 1) * 6]).try_into().unwrap(),
        (&aabbs[ibvh1 * 6..(ibvh1 + 1) * 6]).try_into().unwrap(),
    ) {
        return;
    }
    let ichild0_0 = bvhnodes[ibvh0 * 3 + 1];
    let ichild0_1 = bvhnodes[ibvh0 * 3 + 2];
    let ichild1_0 = bvhnodes[ibvh1 * 3 + 1];
    let ichild1_1 = bvhnodes[ibvh1 * 3 + 2];
    let is_leaf0 = ichild0_1 == usize::MAX;
    let is_leaf1 = ichild1_1 == usize::MAX;
    if !is_leaf0 && !is_leaf1 {
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_0, ichild1_0, bvhnodes, aabbs,
        );
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_1, ichild1_0, bvhnodes, aabbs,
        );
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_0, ichild1_1, bvhnodes, aabbs,
        );
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_1, ichild1_1, bvhnodes, aabbs,
        );
    } else if !is_leaf0 && is_leaf1 {
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_0, ibvh1, bvhnodes, aabbs,
        );
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ichild0_1, ibvh1, bvhnodes, aabbs,
        );
    } else if is_leaf0 && !is_leaf1 {
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ibvh0, ichild1_0, bvhnodes, aabbs,
        );
        search_with_bvh_between_branches(
            pairs, tri2vtx, vtx2xyz, ibvh0, ichild1_1, bvhnodes, aabbs,
        );
    } else if is_leaf0 && is_leaf1 {
        let i_tri = ichild0_0;
        let j_tri = ichild1_0;
        if let Some((p0, p1)) =
            intersection_of_two_triangles_in_mesh(tri2vtx, vtx2xyz, i_tri, j_tri)
        {
            let itp = IntersectingPair {
                i_tri,
                j_tri,
                p0,
                p1,
            };
            pairs.push(itp);
        } else {
            return;
        }
    }
}

pub fn search_with_bvh_inside_branch<T>(
    tripairs: &mut Vec<IntersectingPair<T>>,
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    ibvh: usize,
    bvhnodes: &[usize],
    aabbs: &[T],
) where
    T: nalgebra::RealField + Copy,
{
    let ichild0 = bvhnodes[ibvh * 3 + 1];
    let ichild1 = bvhnodes[ibvh * 3 + 2];
    if ichild1 == usize::MAX {
        return;
    }
    search_with_bvh_between_branches(
        tripairs, tri2vtx, vtx2xyz, ichild0, ichild1, bvhnodes, aabbs,
    );
    search_with_bvh_inside_branch(tripairs, tri2vtx, vtx2xyz, ichild0, bvhnodes, aabbs);
    search_with_bvh_inside_branch(tripairs, tri2vtx, vtx2xyz, ichild1, bvhnodes, aabbs);
}

pub fn search_brute_force<T>(tri2vtx: &[usize], vtx2xyz: &[T]) -> Vec<IntersectingPair<T>>
where
    T: nalgebra::RealField + Copy + 'static,
{
    let mut pairs: Vec<IntersectingPair<T>> = vec![];
    let num_tri = tri2vtx.len() / 3;
    for i_tri in 0..num_tri {
        for j_tri in i_tri + 1..num_tri {
            if let Some((p0, p1)) =
                intersection_of_two_triangles_in_mesh(tri2vtx, vtx2xyz, i_tri, j_tri)
            {
                let itp = IntersectingPair {
                    i_tri,
                    j_tri,
                    p0,
                    p1,
                };
                pairs.push(itp);
            } else {
                continue;
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use crate::trimesh3_intersection::IntersectingPair;
    use crate::{elem2center, elem2elem};

    #[test]
    fn test0() {
        let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::sphere_yup(1.0, 16, 32);
        let pairs = crate::trimesh3_intersection::search_brute_force(&tri2vtx, &vtx2xyz);
        assert_eq!(pairs.len(), 0);
        let (face2idx, idx2node) = elem2elem::face2node_of_polygon_element(3);
        let tri2tri =
            elem2elem::from_uniform_mesh(&tri2vtx, 3, &face2idx, &idx2node, vtx2xyz.len() / 3);
        let tri2center = elem2center::from_uniform_mesh_as_points(&tri2vtx, 3, &vtx2xyz, 3);
        let bvhnodes =
            crate::bvhnodes_topdown_trimesh3::from_uniform_mesh_with_elem2elem_elem2center(
                &tri2tri,
                3,
                &tri2center,
            );
        let mut aabb = Vec::<f32>::new();
        aabb.resize(bvhnodes.len() / 3 * 6, 0.);
        crate::bvh3::update_aabbs_for_uniform_mesh(
            &mut aabb,
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let mut pairs = Vec::<IntersectingPair<f32>>::new();
        crate::trimesh3_intersection::search_with_bvh_inside_branch(
            &mut pairs, &tri2vtx, &vtx2xyz, 0, &bvhnodes, &aabb,
        );
        assert_eq!(pairs.len(), 0);
    }
}
