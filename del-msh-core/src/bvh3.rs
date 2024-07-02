//! method for 3D Bounding Volume Hierarchy

use num_traits::{AsPrimitive, PrimInt};

/// build aabb for uniform mesh
/// if 'elem2vtx' is None, bvh stores the vertex index directly
/// if 'vtx2xyz1' is Some, compute AABB for Continuous-Collision Detection (CCD)
#[allow(clippy::identity_op)]
pub fn update_aabbs_for_uniform_mesh<Index, Real>(
    aabbs: &mut [Real],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: Option<(&[Index], usize)>,
    vtx2xyz0: &[Real],
    vtx2xyz1: Option<&[Real]>,
) where
    Real: num_traits::Float,
    Index: PrimInt + num_traits::AsPrimitive<usize>,
{
    // aabbs.resize();
    assert_eq!(aabbs.len() / 6, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    assert!(if let Some(vtx2xyz1) = vtx2xyz1 {
        vtx2xyz1.len() == vtx2xyz0.len()
    } else {
        true
    });
    if bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_elem: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let aabb = if let Some((elem2vtx, num_noel)) = elem2vtx {
            // element index is provided
            let aabb0 = del_geo_core::aabb3::from_list_of_vertices(
                &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                vtx2xyz0,
                Real::zero(),
            );
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = del_geo_core::aabb3::from_list_of_vertices(
                    &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                    vtx2xyz1,
                    Real::zero(),
                );
                del_geo_core::aabb3::from_two_aabbs(&aabb0, &aabb1)
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
                del_geo_core::aabb3::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        };
        aabbs[i_bvhnode * 6 + 0..i_bvhnode * 6 + 6].copy_from_slice(&aabb[0..6]);
    } else {
        let i_bvhnode_child0: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let i_bvhnode_child1: usize = bvhnodes[i_bvhnode * 3 + 2].as_();
        // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3 + 0].as_(), i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3 + 0].as_(), i_bvhnode);
        // build right tree
        update_aabbs_for_uniform_mesh::<Index, Real>(
            aabbs,
            i_bvhnode_child0,
            bvhnodes,
            elem2vtx,
            vtx2xyz0,
            vtx2xyz1,
        );
        // build left tree
        update_aabbs_for_uniform_mesh::<Index, Real>(
            aabbs,
            i_bvhnode_child1,
            bvhnodes,
            elem2vtx,
            vtx2xyz0,
            vtx2xyz1,
        );
        let aabb = del_geo_core::aabb3::from_two_aabbs(
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

pub fn aabbs_from_uniform_mesh<Index, Real>(
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: Option<(&[Index], usize)>,
    vtx2xyz0: &[Real],
    vtx2xyz1: Option<&[Real]>,
) -> Vec<Real>
where
    Real: num_traits::Float,
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
{
    let num_bvhnode = bvhnodes.len() / 3;
    let mut aabbs = vec![Real::zero(); num_bvhnode * 6];
    update_aabbs_for_uniform_mesh::<Index, Real>(
        &mut aabbs, i_bvhnode, bvhnodes, elem2vtx, vtx2xyz0, vtx2xyz1,
    );
    aabbs
}

pub struct TriMeshWithBvh<'a, Index> {
    pub tri2vtx: &'a [Index],
    pub vtx2xyz: &'a [f32],
    pub bvhnodes: &'a [Index],
    pub aabbs: &'a [f32],
}

pub fn search_intersection_ray<Index>(
    hits: &mut Vec<(f32, usize)>,
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimesh3: &TriMeshWithBvh<Index>,
    i_bvhnode: usize,
) where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    if !del_geo_core::aabb::is_intersect_ray::<3, 6>(
        trimesh3.aabbs[i_bvhnode * 6..i_bvhnode * 6 + 6]
            .try_into()
            .unwrap(),
        ray_org,
        ray_dir,
    ) {
        return;
    }
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.aabbs.len() / 6);
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        // leaf node
        let i_tri: usize = trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_();
        let i0 = trimesh3.tri2vtx[i_tri * 3].as_();
        let i1 = trimesh3.tri2vtx[i_tri * 3 + 1].as_();
        let i2 = trimesh3.tri2vtx[i_tri * 3 + 2].as_();
        let p0 = crate::vtx2xyz::to_array3(trimesh3.vtx2xyz, i0);
        let p1 = crate::vtx2xyz::to_array3(trimesh3.vtx2xyz, i1);
        let p2 = crate::vtx2xyz::to_array3(trimesh3.vtx2xyz, i2);
        let Some(t) =
            del_geo_core::tri3::ray_triangle_intersection_(ray_org, ray_dir, &p0, &p1, &p2)
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
        trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_(),
    );
    search_intersection_ray(
        hits,
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 2].as_(),
    );
}
