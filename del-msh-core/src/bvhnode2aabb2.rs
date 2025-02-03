//! methods related to array of Axis-Aligned Bounding Box (AABB)

use num_traits::AsPrimitive;

/// build aabb for uniform mesh
/// if 'elem2vtx' is None, bvh stores the vertex index directly
/// if 'vtx2xyz1' is Some, compute AABB for Continuous-Collision Detection (CCD)
pub fn update_for_uniform_mesh_with_bvh<Index, Real>(
    bvhnode2aabb: &mut [Real],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: Option<(&[Index], usize)>,
    vtx2xy0: &[Real],
    vtx2xy1: Option<&[Real]>,
) where
    Real: num_traits::Float,
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    use del_geo_core::vec2::Vec2;
    assert_eq!(bvhnode2aabb.len() / 4, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    assert!(if let Some(vtx2xyz1) = vtx2xy1 {
        vtx2xyz1.len() == vtx2xy0.len()
    } else {
        true
    });
    let i_bvhnode_child0 = bvhnodes[i_bvhnode * 3 + 1];
    let i_bvhnode_child1 = bvhnodes[i_bvhnode * 3 + 2];
    if i_bvhnode_child1 == Index::max_value() {
        // leaf node
        let i_elem: usize = i_bvhnode_child0.as_();
        let aabb = if let Some((elem2vtx, num_noel)) = elem2vtx {
            // element index is provided
            let aabb0 = crate::vtx2xy::aabb2_indexed(
                &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                vtx2xy0,
                Real::zero(),
            );
            if let Some(vtx2xyz1) = vtx2xy1 {
                let aabb1 = crate::vtx2xy::aabb2_indexed(
                    &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                    vtx2xyz1,
                    Real::zero(),
                );
                del_geo_core::aabb2::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        } else {
            // no elements. vertex direct
            let aabb0 = crate::vtx2xy::to_vec2(vtx2xy0, i_elem).aabb();
            if let Some(vtx2xy1) = vtx2xy1 {
                let aabb1 = crate::vtx2xy::to_vec2(vtx2xy1, i_elem).aabb();
                del_geo_core::aabb2::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        };
        bvhnode2aabb[i_bvhnode * 4..i_bvhnode * 4 + 4].copy_from_slice(&aabb[0..4]);
    } else {
        let i_bvhnode_child0: usize = i_bvhnode_child0.as_().as_();
        let i_bvhnode_child1: usize = i_bvhnode_child1.as_().as_();
        // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3].as_(), i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3].as_(), i_bvhnode);
        // build right tree
        update_for_uniform_mesh_with_bvh(
            bvhnode2aabb,
            i_bvhnode_child0,
            bvhnodes,
            elem2vtx,
            vtx2xy0,
            vtx2xy1,
        );
        // build left tree
        update_for_uniform_mesh_with_bvh(
            bvhnode2aabb,
            i_bvhnode_child1,
            bvhnodes,
            elem2vtx,
            vtx2xy0,
            vtx2xy1,
        );
        let aabb = del_geo_core::aabb2::from_two_aabbs(
            arrayref::array_ref![bvhnode2aabb, i_bvhnode_child0 * 4, 4],
            arrayref::array_ref![bvhnode2aabb, i_bvhnode_child1 * 4, 4],
        );
        bvhnode2aabb[i_bvhnode * 4..(i_bvhnode + 1) * 4].copy_from_slice(&aabb);
    }
}

pub fn from_uniform_mesh_with_bvh<Index, Real>(
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: Option<(&[Index], usize)>,
    vtx2xyz0: &[Real],
    vtx2xyz1: Option<&[Real]>,
) -> Vec<Real>
where
    Real: num_traits::Float,
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    let num_bvhnode = bvhnodes.len() / 3;
    let mut bvhnode2aabb = vec![Real::zero(); num_bvhnode * 4];
    update_for_uniform_mesh_with_bvh::<Index, Real>(
        &mut bvhnode2aabb,
        i_bvhnode,
        bvhnodes,
        elem2vtx,
        vtx2xyz0,
        vtx2xyz1,
    );
    bvhnode2aabb
}
