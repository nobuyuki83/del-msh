/// Expands a 10-bit integer into 30 bits
/// by putting two zeros before each bit
/// "1011011111" -> "001000001001000001001001001001"
fn expand_bits(x: u32) -> u32 {
    let x = (x | (x << 16)) & 0x030000FF;
    let x = (x | (x << 8)) & 0x0300F00F;
    let x = (x | (x << 4)) & 0x030C30C3;
    (x | (x << 2)) & 0x09249249
}

/// 10-bits for each coordinate
fn morton_code(x: f32, y: f32, z: f32) -> u32 {
    let ix = (x * 1024_f32).max(0_f32).min(1023_f32) as u32;
    let iy = (y * 1024_f32).max(0_f32).min(1023_f32) as u32;
    let iz = (z * 1024_f32).max(0_f32).min(1023_f32) as u32;
    let ix = expand_bits(ix);
    let iy = expand_bits(iy);
    let iz = expand_bits(iz);
    ix * 4 + iy * 2 + iz
}

#[test]
fn test_morton_code() {
    assert_eq!(morton_code(0., 0., 0.), 0u32);
    assert_eq!(morton_code(0., 0., 1.), 0b001001001001001001001001001001);
    assert_eq!(morton_code(1., 0., 1.), 0b101101101101101101101101101101);
    assert_eq!(morton_code(1., 1., 1.), 0xFFFFFFFF >> 2);
}

pub fn sorted_morten_code(
    idx2vtx: &mut [usize],
    idx2morton: &mut [u32],
    vtx2morton: &mut [u32],
    vtx2xyz: &[f32],
) {
    assert_eq!(idx2vtx.len(), idx2morton.len());
    assert_eq!(idx2vtx.len(), vtx2morton.len());
    assert_eq!(idx2vtx.len(), vtx2xyz.len() / 3);
    vtx2xyz
        .chunks(3)
        .zip(vtx2morton.iter_mut())
        .for_each(|(xyz, m)| *m = morton_code(xyz[0], xyz[1], xyz[2]));
    idx2vtx
        .iter_mut()
        .enumerate()
        .for_each(|(iv, idx)| *idx = iv);
    idx2vtx.sort_by_key(|iv| vtx2morton[*iv]);
    for idx in 0..idx2vtx.len() {
        idx2morton[idx] = vtx2morton[idx2vtx[idx]];
    }
}

#[test]
fn test_sorted_morten_code() {
    let vtx2xyz = vec![1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1.];
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2morton = vec![0u32; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut idx2vtx = vec![0usize; num_vtx];
    sorted_morten_code(&mut idx2vtx, &mut idx2morton, &mut vtx2morton, &vtx2xyz);
    for idx in 0..num_vtx - 1 {
        let jdx = idx + 1;
        assert!(idx2morton[idx] <= idx2morton[jdx]);
    }
}

// ---------------

fn delta(idx0: usize, idx1: usize, idx2morton: &[u32]) -> i64 {
    (idx2morton[idx0] ^ idx2morton[idx1]).leading_zeros().into()
}

#[allow(clippy::identity_op)]
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
    let mc1: u32 = idx2morton[idx1 + 0];
    let mc2: u32 = idx2morton[idx1 + 1];
    if mc0 == mc1 && mc1 == mc2 {
        // for hash value collision
        let mut jmc = idx1 + 1;
        while jmc < num_mc {
            jmc += 1;
            if idx2morton[jmc] != mc1 {
                break;
            }
        }
        return (idx1, jmc - 1);
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
        let jmc = idx1 as i64 + lmax * d;
        if jmc < 0 || jmc >= idx2morton.len() as i64 {
            break;
        }
        if delta(idx1, jmc.try_into().unwrap(), idx2morton) <= delta_min {
            break;
        }
        lmax *= 2;
    }

    //find the other end using binary search
    let l = {
        let mut l = 0;
        let mut t = lmax / 2;
        while t >= 1 {
            let jmc = idx1 as i64 + (l + t) * d;
            if jmc >= 0
                && jmc < idx2morton.len() as i64
                && delta(idx1, jmc as usize, idx2morton) > delta_min
            {
                l += t;
            }
            t /= 2;
        }
        l
    };
    let j = (idx1 as i64 + l * d) as usize;
    if idx1 <= j {
        (idx1, j)
    } else {
        (j, idx1)
    }
}

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
        step = (step + 1) / 2; // half step
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

#[allow(clippy::identity_op)]
pub fn bvhnodes_morton(nodes: &mut [usize], idx2vtx: &[usize], idx2morton: &[u32]) {
    assert_eq!(idx2vtx.len(), idx2morton.len());
    assert!(!idx2morton.is_empty());
    assert_eq!(nodes.len(), (idx2morton.len() * 2 - 1) * 3);
    nodes[0] = usize::MAX;

    let num_branch = idx2morton.len() - 1; // number of branch
    for i_branch in 0..num_branch {
        let range = morton_code_determine_range(idx2morton, i_branch);
        let isplit = morton_code_find_split(idx2morton, range.0, range.1);
        assert_ne!(isplit, usize::MAX);
        if range.0 == isplit {
            let i_bvhnode_a = num_branch + isplit; // leaf node
            nodes[i_branch * 3 + 1] = i_bvhnode_a;
            nodes[i_bvhnode_a * 3 + 0] = i_branch;
            nodes[i_bvhnode_a * 3 + 1] = idx2vtx[isplit];
            nodes[i_bvhnode_a * 3 + 2] = usize::MAX;
        } else {
            let i_bvhnode_a = isplit;
            nodes[i_branch * 3 + 1] = i_bvhnode_a;
            nodes[i_bvhnode_a * 3 + 0] = i_branch;
        }
        // ----
        if range.1 == isplit + 1 {
            // leaf node
            let i_bvhnode_b = num_branch + isplit + 1;
            nodes[i_branch * 3 + 2] = i_bvhnode_b;
            nodes[i_bvhnode_b * 3 + 0] = i_branch;
            nodes[i_bvhnode_b * 3 + 1] = idx2vtx[isplit + 1];
            nodes[i_bvhnode_b * 3 + 2] = usize::MAX;
        } else {
            let i_bvhnode_b = isplit + 1;
            nodes[i_branch * 3 + 2] = i_bvhnode_b;
            nodes[i_bvhnode_b * 3 + 0] = i_branch;
        }
    }
}

#[test]
fn test0() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let num_vtx = 100000;
    let mut vtx2xyz = vec![0f32; 0];
    for _ in 0..num_vtx * 3 {
        vtx2xyz.push(rng.gen::<f32>());
    }
    let mut idx2vtx = vec![0usize; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut vtx2morton = vec![0u32; num_vtx];
    sorted_morten_code(&mut idx2vtx, &mut idx2morton, &mut vtx2morton, &vtx2xyz);
    for idx in 0..num_vtx - 1 {
        let jdx = idx + 1;
        assert!(idx2morton[idx] <= idx2morton[jdx]);
    }
    check_morton_code_range_split(&idx2morton);
    let mut bvhnodes = vec![0usize; (num_vtx * 2 - 1) * 3];
    bvhnodes_morton(&mut bvhnodes, &idx2vtx, &idx2morton);
    let mut vtx2cnt = vec![0usize; num_vtx];
    crate::bvh::mark_child(&mut vtx2cnt, 0, &bvhnodes);
    assert_eq!(vtx2cnt, vec!(1usize; num_vtx));
}
