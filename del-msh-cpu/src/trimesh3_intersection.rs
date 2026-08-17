//! method for finding self-intersection using BVH

pub struct IntersectingPair<T> {
    pub i_tri: usize,
    pub j_tri: usize,
    pub p0: [T; 3],
    pub p1: [T; 3],
}

fn intersection_of_two_triangles_in_mesh<T>(
    tri2vtx: &[[usize; 3]],
    vtx2xyz: &[[T; 3]],
    i_tri: usize,
    j_tri: usize,
) -> Option<([T; 3], [T; 3])>
where
    T: Copy + num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    if i_tri == j_tri {
        return None;
    }
    let node2vtx_i = tri2vtx[i_tri];
    let node2vtx_j = tri2vtx[j_tri];
    let icnt0 = node2vtx_j.iter().any(|&j_vtx| j_vtx == node2vtx_i[0]);
    let icnt1 = node2vtx_j.iter().any(|&j_vtx| j_vtx == node2vtx_i[1]);
    let icnt2 = node2vtx_j.iter().any(|&j_vtx| j_vtx == node2vtx_i[2]);
    let num_shared = icnt0 as u8 + icnt1 as u8 + icnt2 as u8;
    match num_shared {
        0 => del_geo_core::tri3::intersection_against_tri3(
            &vtx2xyz[node2vtx_i[0]],
            &vtx2xyz[node2vtx_i[1]],
            &vtx2xyz[node2vtx_i[2]],
            &vtx2xyz[node2vtx_j[0]],
            &vtx2xyz[node2vtx_j[1]],
            &vtx2xyz[node2vtx_j[2]],
        ),
        1 => {
            // sharing one point
            // compute permutation
            let i_node_shared = if icnt0 {
                0
            } else if icnt1 {
                1
            } else {
                2
            };
            let jcnt0 = node2vtx_i.iter().any(|&i_vtx| i_vtx == node2vtx_j[0]);
            let jcnt1 = node2vtx_i.iter().any(|&i_vtx| i_vtx == node2vtx_j[1]);
            let jcnt2 = node2vtx_i.iter().any(|&i_vtx| i_vtx == node2vtx_j[2]);
            assert_eq!(jcnt0 as u8 + jcnt1 as u8 + jcnt2 as u8, 1);
            let j_node_shared = if jcnt0 {
                0
            } else if jcnt1 {
                1
            } else {
                2
            };
            assert_eq!(node2vtx_i[i_node_shared], node2vtx_j[j_node_shared]);
            del_geo_core::tri3::intersection_against_tri3_sharing_vtx(
                &vtx2xyz[node2vtx_i[(i_node_shared + 1) % 3]],
                &vtx2xyz[node2vtx_i[(i_node_shared + 2) % 3]],
                &vtx2xyz[node2vtx_i[i_node_shared]], // shared vertex
                &vtx2xyz[node2vtx_j[(j_node_shared + 1) % 3]],
                &vtx2xyz[node2vtx_j[(j_node_shared + 2) % 3]],
            )
        }
        2 => None,
        3 => {
            panic!("no pair of different triangle should share all the three corner vertices");
        }
        _ => {
            unreachable!();
        }
    }
}

#[allow(clippy::identity_op)]
pub fn search_with_bvh_between_branches<T>(
    pairs: &mut Vec<IntersectingPair<T>>,
    tri2vtx: &[[usize; 3]],
    vtx2xyz: &[[T; 3]],
    ibvh0: usize,
    ibvh1: usize,
    bvhnodes: &[[usize; 3]],
    aabbs: &[[T; 6]],
) where
    T: num_traits::Float + Copy + std::fmt::Debug + std::fmt::Display,
{
    assert!(ibvh0 < aabbs.len());
    assert!(ibvh1 < aabbs.len());
    if !del_geo_core::aabb3::is_intersect(&aabbs[ibvh0], &aabbs[ibvh1]) {
        return;
    }
    let ichild0_0 = bvhnodes[ibvh0][1];
    let ichild0_1 = bvhnodes[ibvh0][2];
    let ichild1_0 = bvhnodes[ibvh1][1];
    let ichild1_1 = bvhnodes[ibvh1][2];
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
        }
    }
}

pub fn search_with_bvh_inside_branch<T>(
    tripairs: &mut Vec<IntersectingPair<T>>,
    tri2vtx: &[[usize; 3]],
    vtx2xyz: &[[T; 3]],
    ibvh: usize,
    bvhnodes: &[[usize; 3]],
    aabbs: &[[T; 6]],
) where
    T: num_traits::Float + Copy + std::fmt::Display + std::fmt::Debug,
{
    let ichild0 = bvhnodes[ibvh][1];
    let ichild1 = bvhnodes[ibvh][2];
    if ichild1 == usize::MAX {
        return;
    }
    search_with_bvh_between_branches(
        tripairs, tri2vtx, vtx2xyz, ichild0, ichild1, bvhnodes, aabbs,
    );
    search_with_bvh_inside_branch(tripairs, tri2vtx, vtx2xyz, ichild0, bvhnodes, aabbs);
    search_with_bvh_inside_branch(tripairs, tri2vtx, vtx2xyz, ichild1, bvhnodes, aabbs);
}

pub fn search_brute_force<T>(tri2vtx: &[[usize; 3]], vtx2xyz: &[[T; 3]]) -> Vec<IntersectingPair<T>>
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    let mut pairs: Vec<IntersectingPair<T>> = vec![];
    let num_tri = tri2vtx.len();
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
        let trimesh3 = crate::trimesh3_primitive::sphere_yup(1.0, 16, 32);
        let pairs =
            crate::trimesh3_intersection::search_brute_force(&trimesh3.tri2vtx, &trimesh3.vtx2xyz);
        assert_eq!(pairs.len(), 0);
        let (face2idx, idx2node) = elem2elem::face2node_of_polygon_element(3);
        let tri2tri = elem2elem::from_uniform_mesh::<_, 3, 3>(
            &trimesh3.tri2vtx,
            &face2idx,
            &idx2node,
            trimesh3.vtx2xyz.len(),
        );
        let tri2center = elem2center::from_uniform_mesh_as_points::<_, _, 3, 3>(
            &trimesh3.tri2vtx,
            &trimesh3.vtx2xyz,
        );
        let bvhnodes =
            crate::bvhnodes_topdown_trimesh3::from_uniform_mesh_with_elem2elem_elem2center(
                &tri2tri,
                &tri2center,
            );
        let mut aabb = Vec::<[f32; 6]>::new();
        aabb.resize(bvhnodes.len(), [0.; 6]);
        crate::bvhnode2aabb3::update_for_uniform_mesh_with_bvh(
            &mut aabb,
            0,
            &bvhnodes,
            &trimesh3.tri2vtx,
            &trimesh3.vtx2xyz,
            None,
        );
        let mut pairs = Vec::<IntersectingPair<f32>>::new();
        crate::trimesh3_intersection::search_with_bvh_inside_branch(
            &mut pairs,
            &trimesh3.tri2vtx,
            &trimesh3.vtx2xyz,
            0,
            &bvhnodes,
            &aabb,
        );
        assert_eq!(pairs.len(), 0);
    }
}
