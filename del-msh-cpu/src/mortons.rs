use num_traits::AsPrimitive;

fn expand_bits2(x: u32) -> u32 {
    let x = (x | (x << 8)) & 0x00ff00ff;
    let x = (x | (x << 4)) & 0x0f0f0f0f;
    let x = (x | (x << 2)) & 0x33333333;
    (x | (x << 1)) & 0x55555555
}

#[test]
fn test_expand_bits2() {
    assert_eq!(expand_bits2(0b11111111), 0b0101010101010101);
    assert_eq!(expand_bits2(0b10001001), 0b0100000001000001);
}

/// compute morton code for 2D point
/// 10-bits for each coordinate
/// * `x` - float number between 0 and 1
/// * `y` - float number between 0 and 1
fn morton_code2(x: f32, y: f32) -> u32 {
    let ix = (x * 1024_f32).clamp(0_f32, 1023_f32) as u32;
    let iy = (y * 1024_f32).clamp(0_f32, 1023_f32) as u32;
    let ix = expand_bits2(ix);
    let iy = expand_bits2(iy);
    ix * 2 + iy
}

#[test]
fn test_morton_code2() {
    assert_eq!(morton_code2(0., 0.), 0u32); // all zero
    assert_eq!(morton_code2(0., 1.), 0b01010101010101010101);
}

pub fn sorted_morten_code2<Index>(
    idx2vtx: &mut [Index],
    idx2morton: &mut [u32],
    vtx2morton: &mut [u32],
    vtx2xy: &[f32],
    transform_xy2uni: &[f32; 9],
) where
    Index: num_traits::PrimInt + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    assert_eq!(idx2vtx.len(), idx2morton.len());
    assert_eq!(idx2vtx.len(), vtx2morton.len());
    assert_eq!(idx2vtx.len(), vtx2xy.len() / 2);
    vtx2xy
        .chunks(2)
        .zip(vtx2morton.iter_mut())
        .for_each(|(xy, m)| {
            let xy = del_geo_core::mat3_col_major::transform_homogeneous(
                transform_xy2uni,
                &[xy[0], xy[1]],
            )
            .unwrap();
            *m = morton_code2(xy[0], xy[1]);
        });
    idx2vtx
        .iter_mut()
        .enumerate()
        .for_each(|(iv, idx)| *idx = iv.as_());
    idx2vtx.sort_by_key(|iv| vtx2morton[(*iv).as_()]);
    for idx in 0..idx2vtx.len() {
        idx2morton[idx] = vtx2morton[idx2vtx[idx].as_()];
    }
}

// above: 2D related
// ------------------------
// below 3D related

/// Expands a 10-bit integer into 30 bits
/// by putting two zeros before each bit
/// "1011011111" -> "001000001001000001001001001001"
fn expand_bits3(x: u32) -> u32 {
    let x = (x | (x << 16)) & 0x030000FF;
    let x = (x | (x << 8)) & 0x0300F00F;
    let x = (x | (x << 4)) & 0x030C30C3;
    (x | (x << 2)) & 0x09249249
}

#[test]
fn test_expand_bits3() {
    assert_eq!(expand_bits3(0b11111111), 0b001001001001001001001001);
    assert_eq!(expand_bits3(0b10001001), 0b001000000000001000000001);
}

/// compute morton code for 3D point
/// 10-bits for each coordinate
/// * `x` - float number between 0 and 1
/// * `y` - float number between 0 and 1
/// * `z` - float number between 0 and 1
fn morton_code3(x: f32, y: f32, z: f32) -> u32 {
    let ix = (x * 1024_f32).clamp(0_f32, 1023_f32) as u32;
    let iy = (y * 1024_f32).clamp(0_f32, 1023_f32) as u32;
    let iz = (z * 1024_f32).clamp(0_f32, 1023_f32) as u32;
    let ix = expand_bits3(ix);
    let iy = expand_bits3(iy);
    let iz = expand_bits3(iz);
    ix * 4 + iy * 2 + iz
}

#[test]
fn test_morton_code3() {
    assert_eq!(morton_code3(0., 0., 0.), 0u32); // all zero
    assert_eq!(morton_code3(0., 0., 1.), 0b001001001001001001001001001001);
    assert_eq!(morton_code3(1., 0., 1.), 0b101101101101101101101101101101);
    assert_eq!(morton_code3(1., 1., 1.), 0xFFFFFFFF >> 2);
}

// above: 3D related
// --------------------

pub fn sorted_morten_code3<Index>(
    idx2vtx: &mut [Index],
    idx2morton: &mut [u32],
    vtx2morton: &mut [u32],
    vtx2xyz: &[f32],
    transform_xy2uni: &[f32; 16],
) where
    Index: num_traits::PrimInt + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    assert_eq!(idx2vtx.len(), idx2morton.len());
    assert_eq!(idx2vtx.len(), vtx2morton.len());
    assert_eq!(idx2vtx.len(), vtx2xyz.len() / 3);
    vtx2xyz
        .chunks(3)
        .zip(vtx2morton.iter_mut())
        .for_each(|(xyz, m)| {
            let xyz = del_geo_core::mat4_col_major::transform_homogeneous(
                transform_xy2uni,
                &[xyz[0], xyz[1], xyz[2]],
            )
            .unwrap();
            *m = morton_code3(xyz[0], xyz[1], xyz[2])
        });
    idx2vtx
        .iter_mut()
        .enumerate()
        .for_each(|(iv, idx)| *idx = iv.as_());
    idx2vtx.sort_by_key(|iv| vtx2morton[(*iv).as_()]);
    for idx in 0..idx2vtx.len() {
        idx2morton[idx] = vtx2morton[idx2vtx[idx].as_()];
    }
}

#[test]
fn test_sorted_morten_code() {
    let vtx2xyz = vec![1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1.];
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2morton = vec![0u32; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut idx2vtx = vec![0usize; num_vtx];
    sorted_morten_code3(
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
}

// ---------------

fn delta(idx0: usize, idx1: usize, idx2morton: &[u32]) -> i64 {
    (idx2morton[idx0] ^ idx2morton[idx1]).leading_zeros().into()
}

/// coverage of this node
pub fn range_of_binary_radix_tree_node(idx2morton: &[u32], idx1: usize) -> (usize, usize) {
    let num_mc = idx2morton.len();
    assert!(!idx2morton.is_empty());
    if idx1 == 0 {
        return (0, num_mc - 1);
    }
    if idx1 == num_mc - 1 {
        // this is only happen in the assertion by "check_morton_code_range_split"
        return (num_mc - 1, num_mc - 1);
    }
    // ----------------------
    let mc0: u32 = idx2morton[idx1 - 1];
    let mc1: u32 = idx2morton[idx1];
    let mc2: u32 = idx2morton[idx1 + 1];
    if mc0 == mc1 && mc1 == mc2 {
        // for hash value collision
        let mut jdx = idx1 + 1;
        while jdx < num_mc - 1 {
            jdx += 1;
            if idx2morton[jdx] != mc1 {
                return (idx1, jdx - 1);
            }
        }
        return (idx1, jdx);
    }
    // get direction
    // (d==+1) -> imc is left-end, move forward
    // (d==-1) -> imc is right-end, move backward
    let d = delta(idx1, idx1 + 1, idx2morton) - delta(idx1, idx1 - 1, idx2morton);
    let d: i64 = if d > 0 { 1 } else { -1 };

    //compute the upper bound for the length of the range
    let delta_min = delta(idx1, (idx1 as i64 - d) as usize, idx2morton);
    let mut lmax: i64 = 2;
    loop {
        let jdx = idx1 as i64 + lmax * d;
        if jdx < 0 || jdx >= idx2morton.len() as i64 {
            break;
        }
        if delta(idx1, jdx.try_into().unwrap(), idx2morton) <= delta_min {
            break;
        }
        lmax *= 2;
    }

    //find the other end using binary search
    let l = {
        let mut l = 0;
        let mut t = lmax / 2;
        while t >= 1 {
            let jdx = idx1 as i64 + (l + t) * d;
            if jdx >= 0
                && jdx < idx2morton.len() as i64
                && delta(idx1, jdx as usize, idx2morton) > delta_min
            {
                l += t;
            }
            t /= 2;
        }
        l
    };
    let jdx = (idx1 as i64 + l * d) as usize;
    if idx1 <= jdx {
        (idx1, jdx)
    } else {
        (jdx, idx1)
    }
}

pub fn split_of_binary_radix_tree_node(
    idx2morton: &[u32],
    i_mc_start: usize,
    i_mc_end: usize,
) -> usize {
    if i_mc_start == i_mc_end {
        return usize::MAX;
    }

    let mc_start: u32 = idx2morton[i_mc_start];
    let nbitcommon0: u32 = (mc_start ^ idx2morton[i_mc_end]).leading_zeros();

    // handle duplicated morton code
    if nbitcommon0 == 32 {
        return i_mc_start;
    } // sizeof(std::uint32_t)*8

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    let mut i_mc_split: usize = i_mc_start; // initial guess
    assert!(i_mc_start <= i_mc_end);
    let mut step: usize = i_mc_end - i_mc_start;
    while step > 1 {
        step = step.div_ceil(2); // (step + 1) / 2; // half step
        let i_mc_new: usize = i_mc_split + step; // proposed new position
        if i_mc_new >= i_mc_end {
            continue;
        }
        let nbitcommon1: u32 = (mc_start ^ idx2morton[i_mc_new]).leading_zeros();
        if nbitcommon1 > nbitcommon0 {
            i_mc_split = i_mc_new; // accept proposal
        }
    }
    i_mc_split
}

/// check sorted morton codes
/// panic if there is a bug in the sorted morton codes
#[allow(dead_code)]
pub fn check_morton_code_range_split(idx2morton: &[u32]) {
    assert!(!idx2morton.is_empty());
    for ini in 0..idx2morton.len() - 1 {
        let range = range_of_binary_radix_tree_node(idx2morton, ini);
        let isplit = split_of_binary_radix_tree_node(idx2morton, range.0, range.1);
        let range_a = range_of_binary_radix_tree_node(idx2morton, isplit);
        let range_b = range_of_binary_radix_tree_node(idx2morton, isplit + 1);
        assert_eq!(range.0, range_a.0);
        assert_eq!(range.1, range_b.1);
        let last1 = if isplit == range.0 { isplit } else { range_a.1 };
        let first1 = if isplit + 1 == range.1 {
            isplit + 1
        } else {
            range_b.0
        };
        assert_eq!(last1 + 1, first1);
    }
}

pub fn update_sorted_morton_code<Index>(
    idx2tri: &mut [Index],
    idx2morton: &mut [u32],
    tri2morton: &mut [u32],
    vtx2xyz: &[f32],
    num_dim: usize,
) where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    match num_dim {
        2 => {
            let aabb = crate::vtx2xy::aabb2(vtx2xyz);
            let transform_xy2uni =
                del_geo_core::aabb2::to_transformation_world2unit_ortho_preserve_asp(&aabb);
            sorted_morten_code2(idx2tri, idx2morton, tri2morton, vtx2xyz, &transform_xy2uni);
        }
        3 => {
            let aabb = crate::vtx2xyz::aabb3(vtx2xyz, 0f32);
            let transform_xy2uni =
                del_geo_core::mat4_col_major::from_aabb3_fit_into_unit_preserve_asp(&aabb);
            // del_geo_core::mat4_col_major::from_aabb3_fit_into_unit(&aabb);
            sorted_morten_code3(idx2tri, idx2morton, tri2morton, vtx2xyz, &transform_xy2uni);
        }
        _ => {
            panic!();
        }
    }
}

pub fn binary_radix_tree<Index>(idx2morton: &[u32], brt: &mut [Index])
where
    Index: num_traits::PrimInt + 'static + Copy,
    usize: AsPrimitive<Index>,
{
    let num_idx = idx2morton.len();
    assert!(!idx2morton.is_empty());
    assert_eq!(brt.len(), (num_idx - 1) * 3);
    brt[0] = Index::max_value();

    let num_branch = idx2morton.len() - 1; // number of branch
    for i_branch in 0..num_branch {
        let range = crate::mortons::range_of_binary_radix_tree_node(idx2morton, i_branch);
        let isplit = crate::mortons::split_of_binary_radix_tree_node(idx2morton, range.0, range.1);
        assert_ne!(isplit, usize::MAX);
        if range.0 == isplit {
            let i_bvhnode_a = num_branch + isplit; // leaf node
            brt[i_branch * 3 + 1] = i_bvhnode_a.as_();
        } else {
            let i_bvhnode_a = isplit;
            brt[i_branch * 3 + 1] = i_bvhnode_a.as_();
            brt[i_bvhnode_a * 3] = i_branch.as_();
        }
        // ----
        if range.1 == isplit + 1 {
            // leaf node
            let i_bvhnode_b = num_branch + isplit + 1;
            brt[i_branch * 3 + 2] = i_bvhnode_b.as_();
        } else {
            let i_bvhnode_b = isplit + 1;
            brt[i_branch * 3 + 2] = i_bvhnode_b.as_();
            brt[i_bvhnode_b * 3] = i_branch.as_();
        }
    }
}

pub fn check_binary_radix_tree(bnodes: &[u32], idx2morton: &[u32]) {
    let num_vtx = idx2morton.len();
    assert_eq!(bnodes.len(), (num_vtx - 1) * 3);
    pub fn increment_leaf_binary_radix_tree<INDEX>(
        bvhnodes: &[INDEX],
        i_node: usize,
        idx2flag: &mut [usize],
    ) where
        INDEX: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    {
        let num_idx = idx2flag.len();
        assert_eq!(bvhnodes.len(), (num_idx - 1) * 3);
        assert!(i_node < num_idx - 1);
        let i0_node = bvhnodes[i_node * 3 + 1].as_();
        if i0_node >= num_idx - 1 {
            let idx = i0_node - (num_idx - 1);
            idx2flag[idx] += 1;
        } else {
            increment_leaf_binary_radix_tree(bvhnodes, i0_node, idx2flag);
        }
        let i1_node = bvhnodes[i_node * 3 + 2].as_();
        if i1_node >= num_idx - 1 {
            let idx = i1_node - (num_idx - 1);
            idx2flag[idx] += 1;
        } else {
            increment_leaf_binary_radix_tree(bvhnodes, i1_node, idx2flag);
        }
    }

    // check binary radix tree
    let mut idx2flag = vec![0usize; num_vtx];
    increment_leaf_binary_radix_tree(bnodes, 0, &mut idx2flag);
    assert_eq!(idx2flag, vec!(1; num_vtx));
    for i_branch in 0..num_vtx - 1 {
        let i_left = bnodes[i_branch * 3 + 1] as usize;
        let i_split = if i_left >= num_vtx - 1 {
            i_left - (num_vtx - 1)
        } else {
            i_left
        };
        let range = range_of_binary_radix_tree_node(idx2morton, i_branch);
        let i_split0 = split_of_binary_radix_tree_node(idx2morton, range.0, range.1);
        assert_eq!(i_split0, i_split);
    }
    /*
    for i_branch in 0..num_vtx - 1 {
        println!("{} --> {} {} {}", i_branch, bnodes[i_branch*3], bnodes[i_branch*3+1], bnodes[i_branch*3+2]);
    }
     */
}

// max_depth: 10 for 3D, 16 for 2D
fn morton2center<const NDIM: usize>(morton: u32, depth: usize, max_depth: usize) -> [f32; NDIM] {
    let mut key = morton >> ((max_depth - depth) * NDIM);
    let mut center = [0.5f32; NDIM];
    for _i_depth in 0..depth {
        for i_dim in 0..NDIM {
            let j = NDIM - i_dim - 1;
            center[j] += (key & 1) as f32;
            center[j] *= 0.5f32;
            key >>= 1;
        }
    }
    center
}

pub fn bnode2depth(
    bnodes: &[u32],
    idx2morton: &[u32],
    num_dim: usize,
    max_depth: usize,
) -> Vec<u32> {
    let num_idx = idx2morton.len();
    assert_eq!(bnodes.len(), (num_idx - 1) * 3);
    let mut bnode2depth = vec![0u32; num_idx - 1];
    for i_branch in 0..num_idx - 1 {
        let i_left = bnodes[i_branch * 3 + 1];
        let i_split = if i_left >= num_idx as u32 - 1 {
            i_left as usize - (num_idx - 1)
        } else {
            i_left as usize
        };
        let delta_split = (idx2morton[i_split] ^ idx2morton[i_split + 1]).leading_zeros();
        // 3D: morton code use 3x10 bits. 32 - 3x10 = 2. Leading two bits are always zero.
        let offset = u32::BITS as usize - num_dim * max_depth;
        bnode2depth[i_branch] = (delta_split - offset as u32) / num_dim as u32;
    }
    bnode2depth
}

pub fn bnode2onode(bnodes: &[u32], bnode2depth: &[u32]) -> Vec<u32> {
    let num_vtx = bnodes.len() / 3 + 1;
    assert_eq!(bnodes.len(), (num_vtx - 1) * 3);
    let mut bnode2isonode = vec![0; num_vtx - 1];
    for i_bnode in 0..num_vtx - 1 {
        if i_bnode == 0 {
            continue;
        }
        let i_bnode_parent = bnodes[i_bnode * 3] as usize;
        if bnode2depth[i_bnode] != bnode2depth[i_bnode_parent] {
            bnode2isonode[i_bnode] = 1;
        }
    }
    // dbg!(&bnode2isonode);
    let mut bnode2onode = vec![0u32; num_vtx - 1];
    // prefix sum
    bnode2onode[0] = bnode2isonode[0];
    for i_bnode in 1..num_vtx - 1 {
        bnode2onode[i_bnode] = bnode2onode[i_bnode - 1] + bnode2isonode[i_bnode];
    }
    bnode2onode
}

pub fn make_octree_from_binary_radix_tree<const NDIM: usize>(
    bnodes: &[u32],
    bnode2onode: &[u32],
    bnode2depth: &[u32],
    idx2morton: &[u32],
    num_onode: usize,
    max_depth: usize,
) -> (Vec<u32>, Vec<usize>, Vec<f32>, Vec<usize>, Vec<f32>) {
    let num_child = 1 << NDIM; // 8 for 3D
    let nlink = (num_child + 1) as usize; // 9 for 3D, 5 for 2D
    let num_vtx = bnodes.len() / 3 + 1;
    assert_eq!(idx2morton.len(), num_vtx);
    let idx2bnode = {
        let mut idx2bnode = vec![0; num_vtx];
        for i_bnode in 0..num_vtx - 1 {
            if bnodes[i_bnode * 3 + 1] as usize >= num_vtx - 1 {
                let idx = bnodes[i_bnode * 3 + 1] as usize - (num_vtx - 1);
                idx2bnode[idx] = i_bnode;
            }
            if bnodes[i_bnode * 3 + 2] as usize >= num_vtx - 1 {
                let idx = bnodes[i_bnode * 3 + 2] as usize - (num_vtx - 1);
                idx2bnode[idx] = i_bnode;
            }
        }
        idx2bnode
    };
    // dbg!(&idx2bnode);
    let mut onodes = vec![u32::MAX; num_onode * nlink];
    let mut idx2onode = vec![0usize; num_vtx];
    let mut onode2depth = vec![0usize; num_onode];
    let mut onode2center = vec![0f32; num_onode * NDIM];
    let mut idx2center = vec![0f32; num_vtx * NDIM];
    for i in 0..num_vtx {
        {
            // set leaf
            let idx = i;
            let (i_onode_parent, depth_parent) = {
                let mut i_bnode_cur = idx2bnode[idx];
                loop {
                    if i_bnode_cur == 0 || bnode2onode[i_bnode_cur] != bnode2onode[i_bnode_cur - 1]
                    {
                        break (bnode2onode[i_bnode_cur] as usize, bnode2depth[i_bnode_cur]);
                    }
                    i_bnode_cur = bnodes[i_bnode_cur * 3] as usize
                }
            };
            assert!(depth_parent < max_depth as u32);
            let key = idx2morton[idx];
            let i_child =
                (key >> ((max_depth - 1 - depth_parent as usize) * NDIM)) & (num_child - 1);
            // println!("leaf: {} {} {}", i_onode_parent, i_child, depth_parent);
            assert_eq!(
                onodes[i_onode_parent * nlink + 1 + i_child as usize],
                u32::MAX
            );
            //
            onodes[i_onode_parent * nlink + 1 + i_child as usize] = (idx + num_onode) as u32;
            idx2onode[idx] = i_onode_parent;
            let center = morton2center::<NDIM>(idx2morton[idx], max_depth, max_depth);
            for i_dim in 0..NDIM {
                idx2center[idx * NDIM + i_dim] = center[i_dim];
            }
        }
        if i < num_vtx - 1 {
            let i_bnode = i;
            if i_bnode == 0 {
                for i_dim in 0..NDIM {
                    onode2center[i_dim] = 0.5f32;
                }
                continue;
            }
            if bnode2onode[i_bnode - 1] == bnode2onode[i_bnode] {
                continue;
            }
            let (i_onode_parent, depth_parent) = {
                let mut i_bnode_parent = bnodes[i_bnode * 3] as usize;
                loop {
                    if i_bnode_parent == 0
                        || bnode2onode[i_bnode_parent] != bnode2onode[i_bnode_parent - 1]
                    {
                        break (
                            bnode2onode[i_bnode_parent] as usize,
                            bnode2depth[i_bnode_parent] as usize,
                        );
                    }
                    i_bnode_parent = bnodes[i_bnode_parent * 3] as usize;
                }
            };
            assert!(depth_parent < max_depth);
            let morton = idx2morton[i_bnode];
            let i_child = (morton >> ((max_depth - 1 - depth_parent) * NDIM)) & (num_child - 1);
            // println!("branch: {} {} {}", i_onode_parent, i_child, depth_parent);
            assert_eq!(
                onodes[i_onode_parent * nlink + 1 + i_child as usize],
                u32::MAX
            );
            //
            let i_onode = bnode2onode[i_bnode] as usize;
            onodes[i_onode_parent * nlink + 1 + i_child as usize] = i_onode as u32;
            onodes[i_onode * nlink] = i_onode_parent as u32;
            //
            let depth = bnode2depth[i_bnode] as usize;
            let center = morton2center::<NDIM>(morton, depth, max_depth);
            for i_dim in 0..NDIM {
                onode2center[i_onode * NDIM + i_dim] = center[i_dim];
            }
            onode2depth[i_onode] = depth;
        }
    }
    (onodes, onode2depth, onode2center, idx2onode, idx2center)
}

pub fn check_octree<const NDIM: usize>(
    idx2onode: &[usize],
    idx2center: &[f32],
    onodes: &[u32],
    onode2depth: &[usize],
    onode2center: &[f32],
    max_depth: usize,
) {
    let nchild = (1 << NDIM) as usize;
    let nlink = nchild + 1;
    let num_onode = onodes.len() / nlink;
    assert_eq!(onodes.len(), num_onode * nlink);
    let num_vtx = idx2onode.len();
    {
        // check if octree visits all the objects
        fn increment_leaf_octree(
            i_node: usize,
            nlink: usize,
            onodes: &[u32],
            idx2isvisited: &mut [u32],
            num_onode: usize,
            nchild: usize,
        ) {
            for i_child in 0..nchild {
                let i_node_child = onodes[i_node * nlink + 1 + i_child];
                if i_node_child == u32::MAX {
                    continue;
                }
                if i_node_child >= num_onode as u32 {
                    let idx0 = i_node_child as usize - num_onode;
                    idx2isvisited[idx0] += 1;
                } else {
                    increment_leaf_octree(
                        i_node_child as usize,
                        nlink,
                        onodes,
                        idx2isvisited,
                        num_onode,
                        nchild,
                    );
                }
            }
        }
        let mut vtx2isvisited = vec![0u32; num_vtx];
        increment_leaf_octree(0, nlink, onodes, &mut vtx2isvisited, num_onode, nchild);
        assert_eq!(vtx2isvisited, vec!(1; num_vtx));
    }
    {
        // check geometry
        for i_onode in 0..num_onode + num_vtx {
            let depth = if i_onode < num_onode {
                onode2depth[i_onode]
            } else {
                max_depth
            };
            let h = 0.5 / (1 << depth) as f32;
            let center: [f32; NDIM] = if i_onode < num_onode {
                onode2center[i_onode * NDIM..(i_onode + 1) * NDIM]
                    .try_into()
                    .unwrap()
            } else {
                let idx = i_onode - num_onode;
                idx2center[idx * NDIM..(idx + 1) * NDIM].try_into().unwrap()
            };
            let i_onode_parent = if i_onode < num_onode {
                onodes[i_onode * nlink]
            } else {
                let idx = i_onode - num_onode;
                idx2onode[idx] as u32
            };
            if i_onode_parent == u32::MAX {
                continue;
            } // root node
            let depth_parent = onode2depth[i_onode_parent as usize];
            assert!(
                depth > depth_parent,
                "child:({} {}), parent:({} {})",
                i_onode,
                depth,
                i_onode_parent,
                depth_parent
            );
            let h_parent = 0.5 / (1 << depth_parent) as f32;
            let center_parent: [f32; NDIM] = onode2center
                [i_onode_parent as usize * NDIM..(i_onode_parent as usize + 1) * NDIM]
                .try_into()
                .unwrap();
            for i_dim in 0..NDIM {
                let d = (center[i_dim] - center_parent[i_dim]).abs();
                assert!(
                    d + h < h_parent * (1. + 1.0e-7),
                    "child:({} {} {:?}), parent:({} {} {:?})",
                    i_onode,
                    depth,
                    center,
                    i_onode_parent,
                    depth_parent,
                    center_parent
                );
            }
        }
    }
}

#[test]
fn test_octree_2d() {
    let num_vtx = 1000usize;
    const NDIM: usize = 2;
    let max_depth = 16;
    let vtx2xyz: Vec<f32> = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(3);
        (0..num_vtx * NDIM).map(|_| rng.random::<f32>()).collect()
    };
    let mut idx2vtx = vec![0usize; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut vtx2morton = vec![0u32; num_vtx];
    sorted_morten_code2(
        &mut idx2vtx,
        &mut idx2morton,
        &mut vtx2morton,
        &vtx2xyz,
        &del_geo_core::mat3_col_major::from_identity(),
    );
    for idx in 0..num_vtx - 1 {
        let jdx = idx + 1;
        assert!(idx2morton[idx] <= idx2morton[jdx]);
    }
    check_morton_code_range_split(&idx2morton);
    // bvh creation
    let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
    binary_radix_tree(&idx2morton, &mut bnodes);
    check_binary_radix_tree(&bnodes, &idx2morton);
    let bnode2depth = bnode2depth(&bnodes, &idx2morton, NDIM, max_depth);
    // dbg!(&bnode2depth);
    let bnode2onode = bnode2onode(&bnodes, &bnode2depth);
    // dbg!(&bnode2onode);
    let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
    println!("num octree node branch:{}", num_onode);
    let (onodes, onode2depth, onode2center, idx2onode, idx2center) =
        make_octree_from_binary_radix_tree::<NDIM>(
            &bnodes,
            &bnode2onode,
            &bnode2depth,
            &idx2morton,
            num_onode,
            max_depth,
        );
    check_octree::<NDIM>(
        &idx2onode,
        &idx2center,
        &onodes,
        &onode2depth,
        &onode2center,
        max_depth,
    );
    // check
}

#[test]
fn test_octree_3d() {
    let num_vtx = 3000usize;
    const NDIM: usize = 3;
    let max_depth = 10;
    let vtx2xyz: Vec<f32> = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        (0..num_vtx * NDIM as usize)
            .map(|_| rng.random::<f32>())
            .collect()
    };
    let mut idx2vtx = vec![0usize; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut vtx2morton = vec![0u32; num_vtx];
    sorted_morten_code3(
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
    check_morton_code_range_split(&idx2morton);
    // bvh creation
    let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
    binary_radix_tree(&idx2morton, &mut bnodes);
    check_binary_radix_tree(&bnodes, &idx2morton);
    let bnode2depth = bnode2depth(&bnodes, &idx2morton, NDIM, max_depth);
    let bnode2onode = bnode2onode(&bnodes, &bnode2depth);
    let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
    println!("num octree node branch:{}", num_onode);
    let (onodes, onode2depth, onode2center, idx2onode, idx2center) =
        make_octree_from_binary_radix_tree::<NDIM>(
            &bnodes,
            &bnode2onode,
            &bnode2depth,
            &idx2morton,
            num_onode,
            max_depth,
        );
    check_octree::<NDIM>(
        &idx2onode,
        &idx2center,
        &onodes,
        &onode2depth,
        &onode2center,
        max_depth,
    );
    // check
}
