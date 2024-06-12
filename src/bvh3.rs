//! method for 3D Bounding Volume Hierarchy

// todo vtx2xyz1: Option<&[Real]>

/// build aabb for uniform mesh
/// if 'elem2vtx' is empty, bvh stores the vertex index directly
/// if 'vtx2xyz1' is not empty, compute AABB for Continuous-Collision Detection (CCD)
#[allow(clippy::identity_op)]
pub fn build_geometry_aabb_for_uniform_mesh<Real>(
    aabbs: &mut [Real],
    i_bvhnode: usize,
    bvhnodes: &[usize],
    elem2vtx: &[usize],
    num_noel: usize,
    vtx2xyz0: &[Real],
    vtx2xyz1: &[Real],
) where
    Real: num_traits::Float,
{
    // aabbs.resize();
    assert_eq!(aabbs.len() / 6, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    assert!(if !vtx2xyz1.is_empty() {
        vtx2xyz1.len() == vtx2xyz0.len()
    } else {
        true
    });
    let i_bvhnode_child0 = bvhnodes[i_bvhnode * 3 + 1];
    let i_bvhnode_child1 = bvhnodes[i_bvhnode * 3 + 2];
    if i_bvhnode_child1 == usize::MAX {
        // leaf node
        let i_elem = i_bvhnode_child0;
        let aabb = if !elem2vtx.is_empty() {
            // element index is provided
            let aabb0 = del_geo::aabb3::from_list_of_vertices(
                &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                vtx2xyz0,
                Real::zero(),
            );
            if vtx2xyz1.is_empty() {
                aabb0
            } else {
                let aabb1 = del_geo::aabb3::from_list_of_vertices(
                    &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                    vtx2xyz1,
                    Real::zero(),
                );
                del_geo::aabb3::from_two_aabbs_slice6(&aabb0, &aabb1)
            }
        } else {
            let aabb0 = [
                vtx2xyz0[i_elem * 3 + 0],
                vtx2xyz0[i_elem * 3 + 1],
                vtx2xyz0[i_elem * 3 + 2],
                vtx2xyz0[i_elem * 3 + 0],
                vtx2xyz0[i_elem * 3 + 1],
                vtx2xyz0[i_elem * 3 + 2],
            ];
            if vtx2xyz1.is_empty() {
                aabb0
            } else {
                let aabb1 = [
                    vtx2xyz1[i_elem * 3 + 0],
                    vtx2xyz1[i_elem * 3 + 1],
                    vtx2xyz1[i_elem * 3 + 2],
                    vtx2xyz1[i_elem * 3 + 0],
                    vtx2xyz1[i_elem * 3 + 1],
                    vtx2xyz1[i_elem * 3 + 2],
                ];
                del_geo::aabb3::from_two_aabbs_slice6(&aabb0, &aabb1)
            }
        };
        aabbs[i_bvhnode * 6 + 0..i_bvhnode * 6 + 6].copy_from_slice(&aabb[0..6]);
    } else {
        // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3 + 0], i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3 + 0], i_bvhnode);
        // build right tree
        build_geometry_aabb_for_uniform_mesh(
            aabbs,
            i_bvhnode_child0,
            bvhnodes,
            elem2vtx,
            num_noel,
            vtx2xyz0,
            vtx2xyz1,
        );
        // build left tree
        build_geometry_aabb_for_uniform_mesh(
            aabbs,
            i_bvhnode_child1,
            bvhnodes,
            elem2vtx,
            num_noel,
            vtx2xyz0,
            vtx2xyz1,
        );
        let aabb = del_geo::aabb3::from_two_aabbs_slice6(
            (&aabbs[i_bvhnode_child0 * 6..(i_bvhnode_child0 + 1) * 6])
                .try_into()
                .unwrap(),
            (&aabbs[i_bvhnode_child1 * 6..(i_bvhnode_child1 + 1) * 6])
                .try_into()
                .unwrap(),
        );
        aabbs[i_bvhnode * 6..(i_bvhnode + 1) * 6].copy_from_slice(&aabb);
    }
}

pub fn intersection_ray(
    hits: &mut Vec<(f32, usize)>,
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    i_bvhnode: usize,
    bvhnodes: &[usize],
    aabbs: &[f32],
) {
    if !del_geo::aabb::is_intersect_ray::<3, 6>(
        aabbs[i_bvhnode * 6..i_bvhnode * 6 + 6].try_into().unwrap(),
        ray_org,
        ray_dir,
    ) {
        return;
    }
    assert_eq!(bvhnodes.len() / 3, aabbs.len() / 6);
    if bvhnodes[i_bvhnode * 3 + 2] == usize::MAX {
        // leaf node
        let i_tri = bvhnodes[i_bvhnode * 3 + 1];
        let p0 = del_geo::vec3::to_array_from_vtx2xyz(vtx2xyz, tri2vtx[i_tri * 3 + 0]);
        let p1 = del_geo::vec3::to_array_from_vtx2xyz(vtx2xyz, tri2vtx[i_tri * 3 + 1]);
        let p2 = del_geo::vec3::to_array_from_vtx2xyz(vtx2xyz, tri2vtx[i_tri * 3 + 2]);
        let Some(t) = del_geo::tri3::ray_triangle_intersection_(ray_org, ray_dir, &p0, &p1, &p2)
        else {
            return;
        };
        hits.push((t, i_tri));
        return;
    }
    intersection_ray(
        hits,
        tri2vtx,
        vtx2xyz,
        ray_org,
        ray_dir,
        bvhnodes[i_bvhnode * 3 + 1],
        bvhnodes,
        aabbs,
    );
    intersection_ray(
        hits,
        tri2vtx,
        vtx2xyz,
        ray_org,
        ray_dir,
        bvhnodes[i_bvhnode * 3 + 2],
        bvhnodes,
        aabbs,
    );
}
