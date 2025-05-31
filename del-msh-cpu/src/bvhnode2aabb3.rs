use num_traits::{AsPrimitive, PrimInt};

/// build aabb for uniform mesh
/// if 'elem2vtx' is None, bvh stores the vertex index directly
/// if 'vtx2xyz1' is Some, compute AABB for Continuous-Collision Detection (CCD)
pub fn update_for_uniform_mesh_with_bvh<Index, Real>(
    bvhnode2aabb: &mut [Real],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: Option<(&[Index], usize)>,
    vtx2xyz0: &[Real],
    vtx2xyz1: Option<&[Real]>,
) where
    Real: num_traits::Float,
    Index: PrimInt + AsPrimitive<usize>,
{
    // aabbs.resize();
    assert_eq!(bvhnode2aabb.len() / 6, bvhnodes.len() / 3);
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
            let aabb0 = crate::vtx2xyz::aabb3_indexed(
                &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                vtx2xyz0,
                Real::zero(),
            );
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = crate::vtx2xyz::aabb3_indexed(
                    &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                    vtx2xyz1,
                    Real::zero(),
                );
                del_geo_core::aabb3::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        } else {
            let aabb0 = crate::vtx2xyz::to_xyz(vtx2xyz0, i_elem).aabb();
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = crate::vtx2xyz::to_xyz(vtx2xyz1, i_elem).aabb();
                del_geo_core::aabb3::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        };
        bvhnode2aabb[i_bvhnode * 6..i_bvhnode * 6 + 6].copy_from_slice(&aabb[0..6]);
    } else {
        let i_bvhnode_child0: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let i_bvhnode_child1: usize = bvhnodes[i_bvhnode * 3 + 2].as_();
        // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3].as_(), i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3].as_(), i_bvhnode);
        // build right tree
        update_for_uniform_mesh_with_bvh::<Index, Real>(
            bvhnode2aabb,
            i_bvhnode_child0,
            bvhnodes,
            elem2vtx,
            vtx2xyz0,
            vtx2xyz1,
        );
        // build left tree
        update_for_uniform_mesh_with_bvh::<Index, Real>(
            bvhnode2aabb,
            i_bvhnode_child1,
            bvhnodes,
            elem2vtx,
            vtx2xyz0,
            vtx2xyz1,
        );
        let aabb = del_geo_core::aabb3::from_two_aabbs(
            arrayref::array_ref!(bvhnode2aabb, i_bvhnode_child0 * 6, 6),
            arrayref::array_ref!(bvhnode2aabb, i_bvhnode_child1 * 6, 6),
        );
        bvhnode2aabb[i_bvhnode * 6..(i_bvhnode + 1) * 6].copy_from_slice(&aabb);
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
    Index: PrimInt + AsPrimitive<usize>,
{
    let num_bvhnode = bvhnodes.len() / 3;
    let mut bvhnode2aabb = vec![Real::zero(); num_bvhnode * 6];
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
