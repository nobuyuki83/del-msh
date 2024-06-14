/// mark child of bvh nodes for assertion purpose
/// `vtx2flag` will be incremented
pub fn mark_child(vtx2flag: &mut [usize], inode0: usize, bvhnodes: &[usize]) {
    assert!(inode0 < bvhnodes.len() / 3);
    if bvhnodes[inode0 * 3 + 2] == usize::MAX {
        // leaf
        let in0 = bvhnodes[inode0 * 3 + 1];
        assert!(in0 < vtx2flag.len());
        vtx2flag[in0] += 1;
        return;
    }
    let in0 = bvhnodes[inode0 * 3 + 1];
    let in1 = bvhnodes[inode0 * 3 + 2];
    mark_child(vtx2flag, in0, bvhnodes);
    mark_child(vtx2flag, in1, bvhnodes);
}

pub fn check_bvh_topology(bvhnodes: &[usize], num_vtx: usize) {
    assert_eq!(bvhnodes.len() % 3, 0);
    let mut vtx2cnt = vec![0usize; num_vtx];
    crate::bvh::mark_child(&mut vtx2cnt, 0, bvhnodes);
    assert_eq!(vtx2cnt, vec!(1usize; num_vtx));
}
