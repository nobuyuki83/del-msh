//! computing BVH using the Linear BVH method (i.e., binary tree based on the Morton code)
//!  * `bvhnodes` - array of index
//!
//!  when the number of query object is N, `bvhnodes` is sized as (2N-1)*3
//!  bvhnodes store the `parent node`, `left node` and `right node` index
//!  if `right node` index is the maximum numbe, the left node stores the index of an object

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

fn morton_code_determine_range(idx2morton: &[u32], idx1: usize) -> (usize, usize) {
    let num_mc = idx2morton.len();
    assert!(!idx2morton.is_empty());
    if idx1 == 0 {
        return (0, num_mc - 1);
    }
    if idx1 == num_mc - 1 {
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

/// check sorted morton codes
/// panic if there is a bug in the sorted morton codes
#[allow(dead_code)]
fn check_morton_code_range_split(idx2morton: &[u32]) {
    assert!(!idx2morton.is_empty());
    for ini in 0..idx2morton.len() - 1 {
        let range = morton_code_determine_range(idx2morton, ini);
        let isplit = morton_code_find_split(idx2morton, range.0, range.1);
        let range_a = morton_code_determine_range(idx2morton, isplit);
        let range_b = morton_code_determine_range(idx2morton, isplit + 1);
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

fn morton_code_find_split(idx2morton: &[u32], i_mc_start: usize, i_mc_end: usize) -> usize {
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
        let range = morton_code_determine_range(idx2morton, i_branch);
        let isplit = morton_code_find_split(idx2morton, range.0, range.1);
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

#[test]
fn test_3d() {
    let num_vtx = 100000;
    let vtx2xyz: Vec<f32> = {
        use rand::Rng;
        let mut rng = rand::rng();
        (0..num_vtx * 3).map(|_| rng.random::<f32>()).collect()
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
    let mut bvhnodes = vec![0usize; (num_vtx * 2 - 1) * 3];
    update_bvhnodes(&mut bvhnodes, &idx2vtx, &idx2morton);
    crate::bvhnodes::check_bvh_topology(&bvhnodes, num_vtx);
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
    sorted_morten_code2(
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
    check_morton_code_range_split(&idx2morton);
    let mut bvhnodes = vec![0usize; (num_vtx * 2 - 1) * 3];
    update_bvhnodes(&mut bvhnodes, &idx2vtx, &idx2morton);
    crate::bvhnodes::check_bvh_topology(&bvhnodes, num_vtx);
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
    update_sorted_morton_code(
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
    update_sorted_morton_code(
        &mut idx2tri,
        &mut idx2morton,
        &mut tri2morton,
        &tri2cntr,
        num_dim,
    );
    update_bvhnodes(bvhnodes, &idx2tri, &idx2morton);
}
