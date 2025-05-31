use num_traits::AsPrimitive;

pub fn including_point<Real, Index>(
    hits: &mut Vec<(Index, Real, Real)>,
    tri2vtx: &[Index],
    vtx2xy: &[Real],
    point: &[Real; 2],
    i_bvhnode: usize,
    bvhnodes: &[Index],
    aabbs: &[Real],
) where
    Real: num_traits::Float,
    Index: AsPrimitive<usize> + num_traits::PrimInt,
    usize: AsPrimitive<Index>,
{
    if !del_geo_core::aabb2::from_aabbs(aabbs, i_bvhnode).is_include_point(point) {
        return;
    }
    assert_eq!(bvhnodes.len() / 3, aabbs.len() / 4);
    if bvhnodes[i_bvhnode * 3 + 2] == Index::max_value() {
        // leaf node
        let i_tri: usize = bvhnodes[i_bvhnode * 3 + 1].as_();
        let Some((r0, r1)) =
            crate::trimesh2::to_tri2(i_tri, tri2vtx, vtx2xy).is_inside(point, Real::one())
        else {
            return;
        };
        hits.push((i_tri.as_(), r0, r1));
        return;
    }
    including_point(
        hits,
        tri2vtx,
        vtx2xy,
        point,
        bvhnodes[i_bvhnode * 3 + 1].as_(),
        bvhnodes,
        aabbs,
    );
    including_point(
        hits,
        tri2vtx,
        vtx2xy,
        point,
        bvhnodes[i_bvhnode * 3 + 2].as_(),
        bvhnodes,
        aabbs,
    );
}
