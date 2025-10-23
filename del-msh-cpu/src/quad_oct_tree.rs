use std::convert::TryInto;

pub fn binary_radix_tree_and_depth<Index>(
    idx2morton: &[u32],
    num_dim: usize,
    max_depth: usize,
    bnodes: &mut [Index],
    bnode2depth: &mut [Index],
) where
    Index: num_traits::PrimInt + 'static + Copy,
    usize: num_traits::AsPrimitive<Index>,
    u32: num_traits::AsPrimitive<Index>,
{
    use num_traits::AsPrimitive;
    let num_idx = idx2morton.len();
    assert!(!idx2morton.is_empty());
    assert_eq!(bnodes.len(), (num_idx - 1) * 3);
    bnodes[0] = Index::max_value();

    let num_bnode = idx2morton.len() - 1; // number of branch
    for i_bnode in 0..num_bnode {
        let range = crate::mortons::range_of_binary_radix_tree_node(idx2morton, i_bnode);
        let i_split = crate::mortons::split_of_binary_radix_tree_node(idx2morton, range.0, range.1);
        assert_ne!(i_split, usize::MAX);
        if range.0 == i_split {
            let i_bvhnode_a = num_bnode + i_split; // leaf node
            bnodes[i_bnode * 3 + 1] = i_bvhnode_a.as_();
        } else {
            let i_bvhnode_a = i_split;
            bnodes[i_bnode * 3 + 1] = i_bvhnode_a.as_();
            bnodes[i_bvhnode_a * 3] = i_bnode.as_();
        }
        // ----
        if range.1 == i_split + 1 {
            // leaf node
            let i_bvhnode_b = num_bnode + i_split + 1;
            bnodes[i_bnode * 3 + 2] = i_bvhnode_b.as_();
        } else {
            let i_bvhnode_b = i_split + 1;
            bnodes[i_bnode * 3 + 2] = i_bvhnode_b.as_();
            bnodes[i_bvhnode_b * 3] = i_bnode.as_();
        }
        // ---
        let delta_split = (idx2morton[i_split] ^ idx2morton[i_split + 1]).leading_zeros();
        // 3D: morton code use 3x10 bits. 32 - 3x10 = 2. Leading two bits are always zero.
        let offset = u32::BITS as usize - num_dim * max_depth;
        bnode2depth[i_bnode] = ((delta_split - offset as u32) / num_dim as u32).as_();
    }
}

// max_depth: 10 for 3D, 16 for 2D
fn morton2center(morton: u32, num_dim: usize, depth: usize, max_depth: usize, center: &mut [f32]) {
    let mut key = morton >> ((max_depth - depth) * num_dim);
    for i_dim in 0..num_dim {
        center[i_dim] = 0.5;
    }
    for _i_depth in 0..depth {
        for i_dim in 0..num_dim {
            let j = num_dim - i_dim - 1;
            center[j] += (key & 1) as f32;
            center[j] *= 0.5f32;
            key >>= 1;
        }
    }
}

pub fn bnode2onode_and_idx2bnode(
    bnodes: &[u32],
    bnode2depth: &[u32],
    bnode2onode: &mut [u32],
    idx2bnode: &mut [u32],
) {
    let num_bnode = bnodes.len() / 3;
    let num_vtx = num_bnode + 1;
    assert_eq!(bnodes.len(), num_bnode * 3);
    assert_eq!(bnode2onode.len(), num_vtx - 1);
    let mut bnode2isonode = vec![0; num_bnode];
    for i_bnode in 0..num_bnode {
        {
            if bnodes[i_bnode * 3 + 1] as usize >= num_vtx - 1 {
                let idx = bnodes[i_bnode * 3 + 1] as usize - (num_vtx - 1);
                idx2bnode[idx] = i_bnode as u32;
            }
            if bnodes[i_bnode * 3 + 2] as usize >= num_vtx - 1 {
                let idx = bnodes[i_bnode * 3 + 2] as usize - (num_vtx - 1);
                idx2bnode[idx] = i_bnode as u32;
            }
        }
        if i_bnode == 0 {
            continue;
        }
        let i_bnode_parent = bnodes[i_bnode * 3] as usize;
        if bnode2depth[i_bnode] != bnode2depth[i_bnode_parent] {
            bnode2isonode[i_bnode - 1] = 1; // shift for exclusive scan
        }
    }
    // prefix sum (exclusive scan)
    bnode2onode[0] = 0;
    for i_bnode in 1..num_vtx - 1 {
        bnode2onode[i_bnode] = bnode2onode[i_bnode - 1] + bnode2isonode[i_bnode - 1];
    }
}

#[allow(clippy::type_complexity)]
pub fn make_tree_from_binary_radix_tree(
    bnodes: &[u32],
    bnode2onode: &[u32],
    bnode2depth: &[u32],
    idx2bnode: &[u32],
    idx2morton: &[u32],
    num_onode: usize,
    max_depth: usize,
    num_dim: usize,
    onodes: &mut [u32],
    onode2depth: &mut [u32],
    onode2center: &mut [f32],
    idx2onode: &mut [u32],
    idx2center: &mut [f32],
) {
    let num_child = 1 << num_dim; // 8 for 3D
    let nlink = (num_child + 1) as usize; // 9 for 3D, 5 for 2D
    let num_vtx = bnodes.len() / 3 + 1;
    assert_eq!(idx2bnode.len(), num_vtx);
    assert_eq!(idx2morton.len(), num_vtx);
    assert_eq!(onodes.len(), num_onode * nlink);
    assert_eq!(onode2depth.len(), num_onode);
    assert_eq!(onode2center.len(), num_onode * num_dim);
    assert_eq!(idx2onode.len(), num_vtx);
    assert_eq!(idx2center.len(), num_vtx * num_dim);
    for i in 0..num_vtx {
        {
            // set leaf
            let idx = i;
            let (i_onode_parent, depth_parent) = {
                let mut i_bnode_cur = idx2bnode[idx] as usize;
                loop {
                    if i_bnode_cur == 0 || bnode2onode[i_bnode_cur] != bnode2onode[i_bnode_cur - 1]
                    {
                        break (bnode2onode[i_bnode_cur] as usize, bnode2depth[i_bnode_cur]);
                    }
                    i_bnode_cur = bnodes[i_bnode_cur * 3] as usize;
                }
            };
            assert!(depth_parent < max_depth as u32);
            let key = idx2morton[idx];
            let i_child =
                (key >> ((max_depth - 1 - depth_parent as usize) * num_dim)) & (num_child - 1);
            // println!("leaf: {} {} {}", i_onode_parent, i_child, depth_parent);
            assert_eq!(
                onodes[i_onode_parent * nlink + 1 + i_child as usize],
                u32::MAX,
                "duplicated child -- idx {}",
                idx
            );
            //
            onodes[i_onode_parent * nlink + 1 + i_child as usize] = (idx + num_onode) as u32;
            idx2onode[idx] = i_onode_parent as u32;
            let mut center = [0f32; 3];
            morton2center(idx2morton[idx], num_dim, max_depth, max_depth, &mut center);
            for i_dim in 0..num_dim {
                idx2center[idx * num_dim + i_dim] = center[i_dim];
            }
        }
        if i < num_vtx - 1 {
            let i_bnode = i;
            if i_bnode == 0 {
                onode2center[0..num_dim].iter_mut().for_each(|v| *v = 0.5);
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
            let i_child = (morton >> ((max_depth - 1 - depth_parent) * num_dim)) & (num_child - 1);
            // println!("branch: {} {} {}", i_onode_parent, i_child, depth_parent);
            assert_eq!(
                onodes[i_onode_parent * nlink + 1 + i_child as usize],
                u32::MAX,
                "duplicated child -- branch {}",
                i_bnode
            );
            //
            let i_onode = bnode2onode[i_bnode] as usize;
            onodes[i_onode_parent * nlink + 1 + i_child as usize] = i_onode as u32;
            onodes[i_onode * nlink] = i_onode_parent as u32;
            //
            let depth = bnode2depth[i_bnode] as usize;
            let mut center = [0f32; 3];
            morton2center(morton, num_dim, depth, max_depth, &mut center);
            for i_dim in 0..num_dim {
                onode2center[i_onode * num_dim + i_dim] = center[i_dim];
            }
            onode2depth[i_onode] = depth as u32;
        }
    }
}

pub fn check_octree<const NDIM: usize>(
    idx2onode: &[u32],
    idx2center: &[f32],
    onodes: &[u32],
    onode2depth: &[u32],
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
                max_depth as u32
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
                idx2onode[idx]
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
    let num_vtx = 10usize;
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
    crate::mortons::sorted_morten_code2(
        &mut idx2vtx,
        &mut idx2morton,
        &mut vtx2morton,
        &vtx2xyz,
        &del_geo_core::mat3_col_major::from_identity(),
    );
    crate::mortons::check_morton_code_range_split(&idx2morton);
    // bvh creation
    let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
    let mut bnode2depth = vec![0u32; num_vtx - 1];
    binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
    crate::mortons::check_binary_radix_tree(&bnodes, &idx2morton);
    let mut bnode2onode = vec![0u32; num_vtx - 1];
    let mut idx2bnode = vec![u32::MAX; num_vtx];
    bnode2onode_and_idx2bnode(&bnodes, &bnode2depth, &mut bnode2onode, &mut idx2bnode);
    let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
    println!("num octree node branch:{}", num_onode);
    let mut onodes = vec![u32::MAX; num_onode * 5];
    let mut idx2onode = vec![0u32; num_vtx];
    let mut onode2depth = vec![0u32; num_onode];
    let mut onode2center = vec![0f32; num_onode * NDIM];
    let mut idx2center = vec![0f32; num_vtx * NDIM];
    make_tree_from_binary_radix_tree(
        &bnodes,
        &bnode2onode,
        &bnode2depth,
        &idx2bnode,
        &idx2morton,
        num_onode,
        max_depth,
        NDIM,
        &mut onodes,
        &mut onode2depth,
        &mut onode2center,
        &mut idx2onode,
        &mut idx2center,
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
    crate::mortons::sorted_morten_code3(
        &mut idx2vtx,
        &mut idx2morton,
        &mut vtx2morton,
        &vtx2xyz,
        &del_geo_core::mat4_col_major::from_identity(),
    );
    crate::mortons::check_morton_code_range_split(&idx2morton);
    // bvh creation
    let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
    let mut bnode2depth = vec![0u32; num_vtx - 1];
    binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
    crate::mortons::check_binary_radix_tree(&bnodes, &idx2morton);
    let mut bnode2onode = vec![0u32; num_vtx - 1];
    let mut idx2bnode = vec![u32::MAX; num_vtx];
    bnode2onode_and_idx2bnode(&bnodes, &bnode2depth, &mut bnode2onode, &mut idx2bnode);
    let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
    println!("num octree node branch:{}", num_onode);
    let mut onodes = vec![u32::MAX; num_onode * 9];
    let mut idx2onode = vec![0u32; num_vtx];
    let mut onode2depth = vec![0u32; num_onode];
    let mut onode2center = vec![0f32; num_onode * NDIM];
    let mut idx2center = vec![0f32; num_vtx * NDIM];
    make_tree_from_binary_radix_tree(
        &bnodes,
        &bnode2onode,
        &bnode2depth,
        &idx2bnode,
        &idx2morton,
        num_onode,
        max_depth,
        NDIM,
        &mut onodes,
        &mut onode2depth,
        &mut onode2center,
        &mut idx2onode,
        &mut idx2center,
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
