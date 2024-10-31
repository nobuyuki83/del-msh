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

/// return the distance to triangle and triangle index
pub fn first_intersection_ray<Index>(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimesh3: &TriMeshWithBvh<Index>,
    i_bvhnode: usize,
    dis: f32,
) -> Option<(f32, usize)>
where
    Index: PrimInt + AsPrimitive<usize>,
{
    if !del_geo_core::aabb3::from_aabbs(trimesh3.aabbs, i_bvhnode)
        .is_intersect_ray(ray_org, ray_dir)
    {
        return None;
    }
    assert_eq!(trimesh3.bvhnodes.len() / 3, trimesh3.aabbs.len() / 6);
    if trimesh3.bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_();
        return crate::trimesh3::to_tri3(i_tri, trimesh3.tri2vtx, trimesh3.vtx2xyz)
            .intersection_against_ray(ray_org, ray_dir)
            .filter(|&t| dis < t)
            .map(|t| (t, i_tri));
    }

    if let Some((t, i_tri)) = first_intersection_ray(
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 1].as_(),
        dis,
    ) {
        let i_aabb = i_bvhnode * 3 + 2;
        // the intesction point is closer than another bvhnode
        if is_point_closer(&trimesh3.aabbs[i_aabb..i_aabb + 6], ray_org, t) {
            return Some((t, i_tri));
        }
        return None;
    }

    first_intersection_ray(
        ray_org,
        ray_dir,
        trimesh3,
        trimesh3.bvhnodes[i_bvhnode * 3 + 2].as_(),
        dis,
    )
}

/// check if a point alone ray_dir is closer than an aabb
fn is_point_closer(aabb: &[f32], ray_dir: &[f32; 3], t: f32) -> bool {
    // cloesest projection of aabb along ray
    let mut proj_aabb = 0.;
    for i_dim in 0..3 {
        if ray_dir[i_dim].abs() == 0. {
            continue;
        }

        if ray_dir[i_dim] > 0. {
            // min project
            proj_aabb += aabb[i_dim] * ray_dir[i_dim];
        } else {
            // max project
            proj_aabb += aabb[i_dim + 3] * ray_dir[i_dim];
        }
    }

    let sqlen = del_geo_core::vec3::dot(ray_dir, ray_dir);
    t * sqlen < proj_aabb
}

fn get_tri_mesh_with_bvh() {
    //TODO: get trimeshwithbvh struct from trimesh
    todo!()
}

#[test]
fn test_first_intersection_ray() {
    //TODO: test sphere
}
