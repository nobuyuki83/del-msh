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

pub fn vtx2morton_from_vtx2co(
    num_dim: usize,
    vtx2co: &[f32],
    transform_co2unit: &[f32],
    vtx2morton: &mut [u32],
) {
    match num_dim {
        2 => {
            assert_eq!(transform_co2unit.len(), 9);
            let transform_co2unit: &[f32; 9] = arrayref::array_ref![transform_co2unit, 0, 9];
            vtx2co
                .chunks(2)
                .zip(vtx2morton.iter_mut())
                .for_each(|(xy, m)| {
                    let xy = del_geo_core::mat3_col_major::transform_homogeneous(
                        transform_co2unit,
                        &[xy[0], xy[1]],
                    )
                    .unwrap();
                    *m = morton_code2(xy[0], xy[1]);
                });
        }
        3 => {
            assert_eq!(transform_co2unit.len(), 16);
            let transform_co2unit: &[f32; 16] = arrayref::array_ref![transform_co2unit, 0, 16];
            vtx2co
                .chunks(3)
                .zip(vtx2morton.iter_mut())
                .for_each(|(xyz, m)| {
                    let xyz = del_geo_core::mat4_col_major::transform_homogeneous(
                        transform_co2unit,
                        &[xyz[0], xyz[1], xyz[2]],
                    )
                    .unwrap();
                    *m = morton_code3(xyz[0], xyz[1], xyz[2])
                });
        }
        _ => {
            panic!()
        }
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
    let num_vtx = idx2morton.len();
    assert!(!idx2morton.is_empty());
    for idx in 0..num_vtx - 1 {
        let jdx = idx + 1;
        assert!(idx2morton[idx] <= idx2morton[jdx]);
    }
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
