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

/// Build 3D Axis-Aligned Bounding Boxes (AABBs) for all nodes in a BVH tree structure.
///
/// Generates a complete array of 3D AABBs for a BVH tree built over uniform mesh elements.
/// Each AABB encloses the 3D geometry of one BVH node. Recursively computes AABBs from
/// leaf nodes (containing elements/vertices) up to the root.
///
/// # Arguments
/// * `i_bvhnode` - Root BVH node index to start building from (typically 0)
/// * `bvhnodes` - BVH tree structure (3 indices per node: parent, child0, child1)
/// * `elem2vtx` - Optional element-to-vertex mapping (vertex_indices, num_nodes_per_element).
///   If None, BVH directly stores vertex indices (no element indirection)
/// * `vtx2xyz0` - Vertex coordinates at time t=0 (required, 3 values per vertex)
/// * `vtx2xyz1` - Optional vertex coordinates at time t=1 for Continuous-Collision Detection (CCD).
///   If provided, AABBs will enclose geometry swept through the 3D motion
///
/// # Returns
/// * `Vec<Real>` - Flattened 3D AABB array where each node i has 6 values at indices [6*i..6*i+6]
///   representing [x_min, y_min, z_min, x_max, y_max, z_max]
///
/// # Notes
/// * AABBs are computed bottom-up: leaf nodes first, then branch nodes merge child AABBs
/// * For CCD, the AABB represents the union of 3D geometry at both start and end positions
/// * All nodes are processed, even if not reachable from i_bvhnode (complete tree coverage)
/// * This is the 3D variant; see `bvhnode2aabb2.rs` for 2D AABBs
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
    // Calculate total number of BVH nodes (each node stores 3 indices)
    let num_bvhnode = bvhnodes.len() / 3;
    // Allocate 3D AABB storage: 6 values per node (x_min, y_min, z_min, x_max, y_max, z_max)
    let mut bvhnode2aabb = vec![Real::zero(); num_bvhnode * 6];
    // Recursively compute AABBs for entire tree from root node
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

pub fn update_for_polygon_polyhedron_mesh_with_bvh<Index, Real>(
    bvhnode2aabb: &mut [Real],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2idx: &[Index],
    idx2vtx: &[Index],
    vtx2xyz: &[Real],
) where
    Real: num_traits::Float,
    Index: PrimInt + AsPrimitive<usize>,
{
    // aabbs.resize();
    assert_eq!(bvhnode2aabb.len() / 6, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    if bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_elem: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let aabb = {
            let zero = Real::zero();
            let one = Real::one();
            let mut aabb3 = [one, zero, zero, zero, zero, zero];
            for &i_vtx in &idx2vtx[elem2idx[i_elem].as_()..elem2idx[i_elem + 1].as_()] {
                let i_vtx = i_vtx.as_();
                let xyz = vtx2xyz[i_vtx * 3..i_vtx * 3 + 3].try_into().unwrap();
                del_geo_core::aabb3::add_point(&mut aabb3, &xyz, zero);
            }
            aabb3
        };
        bvhnode2aabb[i_bvhnode * 6..i_bvhnode * 6 + 6].copy_from_slice(&aabb[0..6]);
    } else {
        let i_bvhnode_child0: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let i_bvhnode_child1: usize = bvhnodes[i_bvhnode * 3 + 2].as_();
        // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3].as_(), i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3].as_(), i_bvhnode);
        // build right tree
        update_for_polygon_polyhedron_mesh_with_bvh::<Index, Real>(
            bvhnode2aabb,
            i_bvhnode_child0,
            bvhnodes,
            elem2idx,
            idx2vtx,
            vtx2xyz,
        );
        // build left tree
        update_for_polygon_polyhedron_mesh_with_bvh::<Index, Real>(
            bvhnode2aabb,
            i_bvhnode_child1,
            bvhnodes,
            elem2idx,
            idx2vtx,
            vtx2xyz,
        );
        let aabb = del_geo_core::aabb3::from_two_aabbs(
            arrayref::array_ref!(bvhnode2aabb, i_bvhnode_child0 * 6, 6),
            arrayref::array_ref!(bvhnode2aabb, i_bvhnode_child1 * 6, 6),
        );
        bvhnode2aabb[i_bvhnode * 6..(i_bvhnode + 1) * 6].copy_from_slice(&aabb);
    }
}

pub fn from_polygon_polyhedron_mesh_with_bvh<Index, Real>(
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2idx: &[Index],
    idx2vtx: &[Index],
    vtx2xyz: &[Real],
) -> Vec<Real>
where
    Real: num_traits::Float,
    Index: PrimInt + AsPrimitive<usize>,
{
    // Calculate total number of BVH nodes (each node stores 3 indices with parent, left, and right)
    let num_bvhnode = bvhnodes.len() / 3;
    // Allocate 3D AABB storage: 6 values per node (x_min, y_min, z_min, x_max, y_max, z_max)
    let mut bvhnode2aabb = vec![Real::zero(); num_bvhnode * 6];
    update_for_polygon_polyhedron_mesh_with_bvh(
        &mut bvhnode2aabb,
        i_bvhnode,
        bvhnodes,
        elem2idx,
        idx2vtx,
        vtx2xyz,
    );
    bvhnode2aabb
}
