/// mark child of bvh nodes for assertion purpose
/// `vtx2flag` will be incremented
pub fn mark_child<INDEX>(bvhnodes: &[INDEX], inode0: usize, vtx2flag: &mut [usize])
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
{
    assert!(inode0 < bvhnodes.len() / 3);
    if bvhnodes[inode0 * 3 + 2] == INDEX::max_value() {
        // leaf
        let in0 = bvhnodes[inode0 * 3 + 1];
        let in0: usize = in0.as_();
        assert!(in0 < vtx2flag.len(), "{} {}", in0, vtx2flag.len());
        vtx2flag[in0] += 1;
        return;
    }
    let in0 = bvhnodes[inode0 * 3 + 1].as_();
    let in1 = bvhnodes[inode0 * 3 + 2].as_();
    mark_child(bvhnodes, in0, vtx2flag);
    mark_child(bvhnodes, in1, vtx2flag);
}

pub fn check_bvh_topology<INDEX>(bvhnodes: &[INDEX], num_vtx: usize)
where
    INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize> + std::fmt::Debug,
{
    assert_eq!(bvhnodes.len() % 3, 0);
    let mut vtx2cnt = vec![0usize; num_vtx];
    mark_child(bvhnodes, 0, &mut vtx2cnt);
    assert_eq!(vtx2cnt, vec!(1usize; num_vtx));
    {
        for i_vtx in 0..num_vtx {
            let i_bvhnode = num_vtx - 1 + i_vtx;
            assert_eq!(bvhnodes[i_bvhnode * 3 + 2], INDEX::max_value());
        }
    }
}
