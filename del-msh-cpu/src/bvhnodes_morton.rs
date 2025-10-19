//! computing BVH using the Linear BVH method (i.e., binary tree based on the Morton code)
//!  * `bvhnodes` - array of index
//!
//!  when the number of query object is N, `bvhnodes` is sized as (2N-1)*3
//!  bvhnodes store the `parent node`, `left node` and `right node` index
//!  if `right node` index is the maximum numbe, the left node stores the index of an object

use num_traits::AsPrimitive;

pub fn update_bvhnodes<Index>(bvhnodes: &mut [Index], idx2vtx: &[Index], idx2morton: &[u32])
where
    Index: num_traits::PrimInt + 'static + Copy,
    usize: AsPrimitive<Index>,
{
    assert_eq!(idx2vtx.len(), idx2morton.len());
    assert!(!idx2morton.is_empty());
    assert_eq!(bvhnodes.len(), (idx2morton.len() * 2 - 1) * 3);
    bvhnodes[0] = Index::max_value();

    let num_branch = idx2morton.len() - 1; // number of branch
    for i_branch in 0..num_branch {
        let range = crate::mortons::range_of_binary_radix_tree_node(idx2morton, i_branch);
        let isplit = crate::mortons::split_of_binary_radix_tree_node(idx2morton, range.0, range.1);
        assert_ne!(isplit, usize::MAX);
        if range.0 == isplit {
            let i_bvhnode_a = num_branch + isplit; // leaf node
            bvhnodes[i_branch * 3 + 1] = i_bvhnode_a.as_();
            bvhnodes[i_bvhnode_a * 3] = i_branch.as_();
            bvhnodes[i_bvhnode_a * 3 + 1] = idx2vtx[isplit];
            bvhnodes[i_bvhnode_a * 3 + 2] = Index::max_value();
        } else {
            let i_bvhnode_a = isplit;
            bvhnodes[i_branch * 3 + 1] = i_bvhnode_a.as_();
            bvhnodes[i_bvhnode_a * 3] = i_branch.as_();
        }
        // ----
        if range.1 == isplit + 1 {
            // leaf node
            let i_bvhnode_b = num_branch + isplit + 1;
            bvhnodes[i_branch * 3 + 2] = i_bvhnode_b.as_();
            bvhnodes[i_bvhnode_b * 3] = i_branch.as_();
            bvhnodes[i_bvhnode_b * 3 + 1] = idx2vtx[isplit + 1];
            bvhnodes[i_bvhnode_b * 3 + 2] = Index::max_value();
        } else {
            let i_bvhnode_b = isplit + 1;
            bvhnodes[i_branch * 3 + 2] = i_bvhnode_b.as_();
            bvhnodes[i_bvhnode_b * 3] = i_branch.as_();
        }
    }
}

/*
pub fn update_for_vtx2xyz<Index>(bvhnodes: &mut [Index], vtx2xyz: &[f32], num_dim: usize)
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let num_tri = vtx2xyz.len() / num_dim;
    assert_eq!(bvhnodes.len(), (num_tri * 2 - 1) * 3);
    let mut idx2tri = vec![Index::one(); num_tri];
    let mut idx2morton = vec![0u32; num_tri];
    let mut tri2morton = vec![0u32; num_tri];
    update_for_sorted_morton_code(
        &mut idx2tri,
        &mut idx2morton,
        &mut tri2morton,
        vtx2xyz,
        num_dim,
    );
    bvhnodes_morton(bvhnodes, &idx2tri, &idx2morton);
}
 */

pub fn from_vtx2xyz<Index>(vtx2xyz: &[f32], num_dim: usize) -> Vec<Index>
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let num_tri = vtx2xyz.len() / num_dim;
    let mut idx2tri = vec![Index::one(); num_tri];
    let mut idx2morton = vec![0u32; num_tri];
    let mut tri2morton = vec![0u32; num_tri];
    crate::mortons::update_sorted_morton_code(
        &mut idx2tri,
        &mut idx2morton,
        &mut tri2morton,
        vtx2xyz,
        num_dim,
    );
    let mut bvhnodes = vec![Index::zero(); (num_tri * 2 - 1) * 3];
    update_bvhnodes(&mut bvhnodes, &idx2tri, &idx2morton);
    bvhnodes
}

pub fn from_triangle_mesh<Index>(tri2vtx: &[Index], vtx2xy: &[f32], num_dim: usize) -> Vec<Index>
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let tri2cntr =
        crate::elem2center::from_uniform_mesh_as_points::<Index, f32>(tri2vtx, 3, vtx2xy, num_dim);
    from_vtx2xyz(&tri2cntr, num_dim)
}

pub fn update_for_triangle_mesh<Index>(
    bvhnodes: &mut [Index],
    tri2vtx: &[Index],
    vtx2xy: &[f32],
    num_dim: usize,
) where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let num_tri = tri2vtx.len() / 3;
    assert_eq!(bvhnodes.len(), (num_tri * 2 - 1) * 3);
    let tri2cntr =
        crate::elem2center::from_uniform_mesh_as_points::<Index, f32>(tri2vtx, 3, vtx2xy, num_dim);
    let mut idx2tri = vec![Index::one(); num_tri];
    let mut idx2morton = vec![0u32; num_tri];
    let mut tri2morton = vec![0u32; num_tri];
    crate::mortons::update_sorted_morton_code(
        &mut idx2tri,
        &mut idx2morton,
        &mut tri2morton,
        &tri2cntr,
        num_dim,
    );
    update_bvhnodes(bvhnodes, &idx2tri, &idx2morton);
}

#[test]
fn test_2d() {
    let num_vtx = 100000;
    let vtx2xy: Vec<f32> = {
        use rand::Rng;
        let mut rng = rand::rng();
        (0..num_vtx * 2).map(|_| rng.random::<f32>()).collect()
    };
    let mut idx2vtx = vec![0usize; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut vtx2morton = vec![0u32; num_vtx];
    crate::mortons::sorted_morten_code2(
        &mut idx2vtx,
        &mut idx2morton,
        &mut vtx2morton,
        &vtx2xy,
        &del_geo_core::mat3_col_major::from_identity::<f32>(),
    );
    for idx in 0..num_vtx - 1 {
        let jdx = idx + 1;
        assert!(idx2morton[idx] <= idx2morton[jdx]);
    }
    crate::mortons::check_morton_code_range_split(&idx2morton);
    let mut bvhnodes = vec![0usize; (num_vtx * 2 - 1) * 3];
    crate::bvhnodes_morton::update_bvhnodes(&mut bvhnodes, &idx2vtx, &idx2morton);
    crate::bvhnodes::check_bvh_topology(&bvhnodes, num_vtx);
}
#[test]
fn test_3d() {
    let num_vtx = 30000;
    let num_dim = 3;
    let vtx2xyz: Vec<f32> = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        (0..num_vtx * num_dim)
            .map(|_| rng.random::<f32>())
            .collect()
    };
    let mut idx2vtx = vec![0usize; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut vtx2morton = vec![0u32; num_vtx];
    crate::mortons::sorted_morten_code3(
        &mut idx2vtx,
        &mut idx2morton,
        &mut vtx2morton,
        &vtx2xyz,
        &del_geo_core::mat4_col_major::from_identity(),
    );
    for idx in 0..num_vtx - 1 {
        let jdx = idx + 1;
        assert!(idx2morton[idx] <= idx2morton[jdx]);
    }
    crate::mortons::check_morton_code_range_split(&idx2morton);
    // bvh creation
    let mut bvhnodes = vec![0usize; (num_vtx * 2 - 1) * 3];
    update_bvhnodes(&mut bvhnodes, &idx2vtx, &idx2morton);
    crate::bvhnodes::check_bvh_topology(&bvhnodes, num_vtx);
}
