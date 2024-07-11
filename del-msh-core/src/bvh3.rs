//! method for 3D Bounding Volume Hierarchy

use num_traits::{AsPrimitive, PrimInt};

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
    Index: PrimInt + AsPrimitive<usize>,
{
    if !del_geo_core::aabb3::from_aabbs(trimesh3.aabbs, i_bvhnode)
        .is_intersect_ray(ray_org, ray_dir)
    {
        return;
    }
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.aabbs.len() / 6);
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_();
        let Some(t) = crate::trimesh3::to_tri3(i_tri, trimesh3.tri2vtx, trimesh3.vtx2xyz)
            .intersection_against_ray(ray_org, ray_dir)
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
