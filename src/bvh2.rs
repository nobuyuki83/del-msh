use num_traits::AsPrimitive;

/// build aabb for uniform mesh
/// if 'elem2vtx' is empty, bvh stores the vertex index directly
/// if 'vtx2xyz1' is not empty, compute AABB for Continuous-Collision Detection (CCD)
#[allow(clippy::identity_op)]
pub fn update_aabbs_for_uniform_mesh<Index, Real>(
    aabbs: &mut [Real],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: &[Index],
    num_noel: usize,
    vtx2xyz0: &[Real],
    vtx2xyz1: Option<&[Real]>,
) where
    Real: num_traits::Float,
    Index: num_traits::PrimInt + AsPrimitive<usize>
{
    assert_eq!(aabbs.len() / 4, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    assert!(if let Some(vtx2xyz1) = vtx2xyz1 {
        vtx2xyz1.len() == vtx2xyz0.len()
    } else {
        true
    });
    let i_bvhnode_child0 = bvhnodes[i_bvhnode * 3 + 1];
    let i_bvhnode_child1 = bvhnodes[i_bvhnode * 3 + 2];
    if i_bvhnode_child1 == Index::max_value() {
        // leaf node
        let i_elem: usize = i_bvhnode_child0.as_();
        let aabb = if !elem2vtx.is_empty() {
            // element index is provided
            let aabb0 = del_geo::aabb2::from_list_of_vertices(
                &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                vtx2xyz0,
                Real::zero(),
            );
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = del_geo::aabb2::from_list_of_vertices(
                    &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
                    vtx2xyz1,
                    Real::zero(),
                );
                del_geo::aabb2::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        } else {
            // vertex direct
            let aabb0 = [
                vtx2xyz0[i_elem * 2 + 0],
                vtx2xyz0[i_elem * 2 + 1],
                vtx2xyz0[i_elem * 2 + 0],
                vtx2xyz0[i_elem * 2 + 1],
            ];
            if let Some(vtx2xyz1) = vtx2xyz1 {
                let aabb1 = [
                    vtx2xyz1[i_elem * 2 + 0],
                    vtx2xyz1[i_elem * 2 + 1],
                    vtx2xyz1[i_elem * 2 + 0],
                    vtx2xyz1[i_elem * 2 + 1],
                ];
                del_geo::aabb2::from_two_aabbs(&aabb0, &aabb1)
            } else {
                aabb0
            }
        };
        aabbs[i_bvhnode * 4 + 0..i_bvhnode * 4 + 4].copy_from_slice(&aabb[0..4]);
    } else {
        let i_bvhnode_child0: usize = i_bvhnode_child0.as_().as_();
        let i_bvhnode_child1: usize = i_bvhnode_child1.as_().as_();
        // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3 + 0].as_(), i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3 + 0].as_(), i_bvhnode);
        // build right tree
        crate::bvh2::update_aabbs_for_uniform_mesh(
            aabbs,
            i_bvhnode_child0,
            bvhnodes,
            elem2vtx,
            num_noel,
            vtx2xyz0,
            vtx2xyz1,
        );
        // build left tree
        crate::bvh2::update_aabbs_for_uniform_mesh(
            aabbs,
            i_bvhnode_child1,
            bvhnodes,
            elem2vtx,
            num_noel,
            vtx2xyz0,
            vtx2xyz1,
        );
        let aabb = del_geo::aabb2::from_two_aabbs(
            (&aabbs[i_bvhnode_child0 * 4..(i_bvhnode_child0 + 1) * 4])
                .try_into()
                .unwrap(),
            (&aabbs[i_bvhnode_child1 * 4..(i_bvhnode_child1 + 1) * 4])
                .try_into()
                .unwrap(),
        );
        aabbs[i_bvhnode * 4..(i_bvhnode + 1) * 4].copy_from_slice(&aabb);
    }
}

pub fn aabbs_from_uniform_mesh<Index, Real>(
    i_bvhnode: usize,
    bvhnodes: &[Index],
    elem2vtx: &[Index],
    num_noel: usize,
    vtx2xyz0: &[Real],
    vtx2xyz1: Option<&[Real]>) -> Vec<Real>
    where
    Real: num_traits::Float,
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>
{
    let num_bvhnode = bvhnodes.len() / 3;
    let mut aabbs = vec!(Real::zero(); num_bvhnode*4);
    update_aabbs_for_uniform_mesh::<Index, Real>(
        &mut aabbs, i_bvhnode, bvhnodes, elem2vtx, num_noel, vtx2xyz0, vtx2xyz1);
    aabbs
}

pub fn search_including_point<Real, Index>(
    hits: &mut Vec<(usize, Real, Real)>,
    tri2vtx: &[Index],
    vtx2xy: &[Real],
    point: &[Real; 2],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    aabbs: &[Real],
) where
    Real: num_traits::Float,
    Index: AsPrimitive<usize> + num_traits::PrimInt
{
    if !del_geo::aabb::is_include_point::<Real, 2, 4>(
        aabbs[i_bvhnode * 4..(i_bvhnode + 1) * 4]
            .try_into()
            .unwrap(),
        point,
    ) {
        return;
    }
    assert_eq!(bvhnodes.len() / 3, aabbs.len() / 4);
    if bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let p0 = del_geo::vec2::to_array_from_vtx2xy(vtx2xy, tri2vtx[i_tri * 3]);
        let p1 = del_geo::vec2::to_array_from_vtx2xy(vtx2xy, tri2vtx[i_tri * 3 + 1]);
        let p2 = del_geo::vec2::to_array_from_vtx2xy(vtx2xy, tri2vtx[i_tri * 3 + 2]);
        let Some((r0, r1)) = del_geo::tri2::is_inside(&p0, &p1, &p2, point, Real::one()) else {
            return;
        };
        hits.push((i_tri, r0, r1));
        return;
    }
    crate::bvh2::search_including_point(
        hits,
        tri2vtx,
        vtx2xy,
        point,
        bvhnodes[i_bvhnode * 3 + 1].as_(),
        bvhnodes,
        aabbs,
    );
    crate::bvh2::search_including_point(
        hits,
        tri2vtx,
        vtx2xy,
        point,
        bvhnodes[i_bvhnode * 3 + 2].as_(),
        bvhnodes,
        aabbs,
    );
}
