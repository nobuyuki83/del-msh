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
    vtx2xyz1: Option<&[Real]>,
) where
    Real: num_traits::Float,
{
    // aabbs.resize();
    assert_eq!(aabbs.len() / 6, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    assert!(if let Some(vtx2xyz1) = vtx2xyz1 {
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
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = del_geo::aabb3::from_list_of_vertices(
                    &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                    vtx2xyz1,
                    Real::zero(),
                );
                del_geo::aabb3::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
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
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = [
                    vtx2xyz1[i_elem * 3 + 0],
                    vtx2xyz1[i_elem * 3 + 1],
                    vtx2xyz1[i_elem * 3 + 2],
                    vtx2xyz1[i_elem * 3 + 0],
                    vtx2xyz1[i_elem * 3 + 1],
                    vtx2xyz1[i_elem * 3 + 2],
                ];
                del_geo::aabb3::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
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
        let aabb = del_geo::aabb3::from_two_aabbs(
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

pub struct TriMesh3Bvh<'a> {
    pub tri2vtx: &'a [usize],
    pub vtx2xyz: &'a [f32],
    pub bvhnodes: &'a [usize],
    pub aabbs: &'a [f32]
}

pub fn search_intersection_ray(
    hits: &mut Vec<(f32, usize)>,
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimesh3: &TriMesh3Bvh,
    i_bvhnode: usize,
) {
    if !del_geo::aabb::is_intersect_ray::<3, 6>(
        trimesh3.aabbs[i_bvhnode * 6..i_bvhnode * 6 + 6].try_into().unwrap(),
        ray_org,
        ray_dir,
    ) {
        return;
    }
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.aabbs.len() / 6);
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == usize::MAX {
        // leaf node
        let i_tri = trimesh3.bvhnodes[i_bvhnode * 3 + 1];
        let p0 = del_geo::vec3::to_array_from_vtx2xyz(trimesh3.vtx2xyz, trimesh3.tri2vtx[i_tri * 3]);
        let p1 = del_geo::vec3::to_array_from_vtx2xyz(trimesh3.vtx2xyz, trimesh3.tri2vtx[i_tri * 3 + 1]);
        let p2 = del_geo::vec3::to_array_from_vtx2xyz(trimesh3.vtx2xyz, trimesh3.tri2vtx[i_tri * 3 + 2]);
        let Some(t) = del_geo::tri3::ray_triangle_intersection_(ray_org, ray_dir, &p0, &p1, &p2)
        else {
            return;
        };
        hits.push((t, i_tri));
        return;
    }
    search_intersection_ray(
        hits,
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 1],
    );
    search_intersection_ray(
        hits,
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 2],
    );
}
