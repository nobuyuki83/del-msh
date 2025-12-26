// max_depth is typically 32 / ndim
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
    assert!(u32::BITS >= (num_dim * max_depth) as u32);
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
    center.iter_mut().take(num_dim).for_each(|v| *v = 0.5);
    for _i_depth in 0..depth {
        for i_dim in 0..num_dim {
            let j = num_dim - i_dim - 1;
            center[j] += (key & 1) as f32;
            center[j] *= 0.5f32;
            key >>= 1;
        }
    }
}

#[test]
fn test_morton2center() {
    {
        let p = [0.321, 0.123];
        let mc = crate::mortons::morton_code2(p[0], p[1]);
        let mut center = [0f32; 3];
        morton2center(mc, 2, 16, 16, &mut center);
        let h = 1. / (1 << 16) as f32;
        assert!((center[0] - p[0]).abs() < h);
        assert!((center[1] - p[1]).abs() < h);
    }
    {
        let p = [0.321, 0.123, 0.87321];
        let mc = crate::mortons::morton_code3(p[0], p[1], p[2]);
        let mut center = [0f32; 3];
        morton2center(mc, 3, 10, 10, &mut center);
        let h = 1. / (1 << 10) as f32;
        assert!((center[0] - p[0]).abs() < h);
        assert!((center[1] - p[1]).abs() < h);
        assert!((center[2] - p[2]).abs() < h);
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
            if bnodes[i_bnode * 3 + 1] as usize >= num_bnode {
                let idx = bnodes[i_bnode * 3 + 1] as usize - num_bnode;
                idx2bnode[idx] = i_bnode as u32;
            }
            if bnodes[i_bnode * 3 + 2] as usize >= num_bnode {
                let idx = bnodes[i_bnode * 3 + 2] as usize - num_bnode;
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
#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
pub fn check_octree_vtx2xyz<const NDIM: usize, const NAFFINE: usize>(
    vtx2xyz: &[f32],
    transform_world2unit: &[f32; NAFFINE],
    idx2jdx_offset: &[u32],
    jdx2vtx: &[u32],
    idx2onode: &[u32],
    idx2center: &[f32],
    max_depth: usize,
    onode2center: &[f32],
    onode2depth: &[u32],
) {
    assert_eq!(NAFFINE, (NDIM + 1) * (NDIM + 1));
    let num_idx = idx2jdx_offset.len() - 1;
    let num_vtx = vtx2xyz.len() / NDIM;
    assert_eq!(vtx2xyz.len(), num_vtx * NDIM);
    assert_eq!(jdx2vtx.len(), num_vtx);
    assert_eq!(idx2onode.len(), num_idx);
    assert_eq!(idx2center.len(), num_idx * NDIM);
    let num_onode = onode2center.len() / NDIM;
    assert_eq!(onode2center.len(), num_onode * NDIM);
    assert_eq!(onode2depth.len(), num_onode);
    for idx in 0..num_idx {
        let i_onode = idx2onode[idx] as usize;
        assert!(i_onode < num_onode);
        let center_cell_unit = &idx2center[idx * NDIM..(idx + 1) * NDIM];
        let h_cell_vtx = 0.5 / (1 << max_depth) as f32;
        for jdx in idx2jdx_offset[idx]..idx2jdx_offset[idx + 1] {
            let i_vtx = jdx2vtx[jdx as usize] as usize;
            let pos_vtx_unit = match NDIM {
                2 => {
                    let pos_vtx_world = arrayref::array_ref![vtx2xyz, i_vtx * NDIM, 2];
                    let transform_world2unit = arrayref::array_ref![transform_world2unit, 0, 9];
                    del_geo_core::mat3_col_major::transform_homogeneous(
                        transform_world2unit,
                        pos_vtx_world,
                    )
                    .unwrap()
                    .to_vec()
                }
                3 => {
                    let pos_vtx_world = arrayref::array_ref![vtx2xyz, i_vtx * NDIM, 3];
                    let transform_world2unit = arrayref::array_ref![transform_world2unit, 0, 16];
                    del_geo_core::mat4_col_major::transform_homogeneous(
                        transform_world2unit,
                        pos_vtx_world,
                    )
                    .unwrap()
                    .to_vec()
                }
                _ => {
                    panic!()
                }
            };
            for i_dim in 0..NDIM {
                // check the vtx position in unit coordinate is inside the cell
                let d = (pos_vtx_unit[i_dim] - center_cell_unit[i_dim]).abs();
                assert!(d <= h_cell_vtx * 1.0001, "{} {}", d, h_cell_vtx);
            }
        }
        let center_cell_parent = &onode2center[i_onode * NDIM..(i_onode + 1) * NDIM];
        let h_cell_parent = 0.5 / (1 << onode2depth[i_onode]) as f32;
        /*
        println!(
            "{:?}, {:?}, {:?}",
            center_cell_unit, pos_vtx_unit, center_cell_parent
        );
         */
        for i_dim in 0..NDIM {
            // check the vtx position in unit coordinate is inside the cell
            let d = (center_cell_parent[i_dim] - center_cell_unit[i_dim]).abs();
            assert!(
                d + h_cell_vtx <= h_cell_parent,
                "{} {}",
                d + h_cell_vtx,
                h_cell_parent
            );
        }
    }
}

pub fn aggregate(
    num_vdim: usize,
    idx2val: &[f32],
    idx2onode: &[u32],
    num_link: usize,
    onodes: &[u32],
    onode2aggval: &mut [f32],
) {
    onode2aggval.fill(0.0);
    let num_onode = onodes.len() / num_link;
    let num_idx = idx2val.len() / num_vdim;
    //
    assert_eq!(onodes.len(), num_onode * num_link);
    assert_eq!(idx2val.len(), num_idx * num_vdim);
    assert_eq!(onode2aggval.len(), num_onode * num_vdim);
    //
    for idx in 0..num_idx {
        let mut i_onode = idx2onode[idx] as usize;
        assert!(i_onode < num_onode);
        loop {
            for i_vdim in 0..num_vdim {
                onode2aggval[i_onode * num_vdim + i_vdim] += idx2val[idx * num_vdim + i_vdim];
            }
            if onodes[i_onode * 9] == u32::MAX {
                break;
            }
            i_onode = onodes[i_onode * 9] as usize;
            assert!(i_onode < num_onode);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn aggregate_with_map(
    idx2jdx_offset: &[u32],
    num_vdim: usize,
    jdx2vtx: &[u32],
    vtx2val: &[f32],
    idx2onode: &[u32],
    num_link: usize,
    onode2idx_tree: &[u32],
    onode2aggval: &mut [f32],
) {
    let num_onode = onode2idx_tree.len() / num_link;
    let num_idx = idx2jdx_offset.len() - 1;
    let num_vtx = jdx2vtx.len();
    //
    assert_eq!(idx2onode.len(), num_idx);
    assert_eq!(vtx2val.len(), num_vtx * num_vdim);
    assert_eq!(onode2idx_tree.len(), num_onode * num_link);
    assert_eq!(onode2aggval.len(), num_onode * num_vdim);
    //
    onode2aggval.fill(0.0);
    for idx in 0..num_idx {
        let mut i_onode = idx2onode[idx] as usize;
        assert!(i_onode < num_onode);
        for jdx in idx2jdx_offset[idx]..idx2jdx_offset[idx + 1] {
            let i_vtx = jdx2vtx[jdx as usize] as usize;
            assert!(i_vtx < num_vtx);
            let val = &vtx2val[i_vtx * num_vdim..(i_vtx + 1) * num_vdim];
            loop {
                for i_vdim in 0..num_vdim {
                    onode2aggval[i_onode * num_vdim + i_vdim] += val[i_vdim];
                }
                if onode2idx_tree[i_onode * 9] == u32::MAX {
                    break;
                }
                i_onode = onode2idx_tree[i_onode * 9] as usize;
                assert!(i_onode < num_onode);
            }
        }
    }
}

pub fn onode2gcuint_for_octree(
    idx2jdx_offset: &[u32],
    jdx2vtx: &[u32],
    idx2onode: &[u32],
    vtx2xyz: &[f32],
    transform_world2unit: &[f32; 16],
    onode2idx_otree: &[u32],
    onode2gcunit: &mut [[f32; 3]],
) {
    let num_idx = idx2jdx_offset.len() - 1;
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(jdx2vtx.len(), num_vtx);
    assert_eq!(vtx2xyz.len(), num_vtx * 3);
    let num_onode = onode2idx_otree.len() / 9;
    assert_eq!(onode2gcunit.len(), num_onode);
    let mut onode2nvtx = vec![0f32; num_onode];
    onode2gcunit.fill([0f32; 3]);
    for idx in 0..num_idx {
        for jdx in idx2jdx_offset[idx]..idx2jdx_offset[idx + 1] {
            let i_vtx = jdx2vtx[jdx as usize] as usize;
            let mut i_onode = idx2onode[idx] as usize;
            assert!(i_onode < num_onode);
            let pos_vtx_world = arrayref::array_ref![vtx2xyz, i_vtx * 3, 3];
            loop {
                onode2gcunit[i_onode][0] += pos_vtx_world[0];
                onode2gcunit[i_onode][1] += pos_vtx_world[1];
                onode2gcunit[i_onode][2] += pos_vtx_world[2];
                onode2nvtx[i_onode] += 1.;
                if onode2idx_otree[i_onode * 9] == u32::MAX {
                    break;
                }
                i_onode = onode2idx_otree[i_onode * 9] as usize;
                assert!(i_onode < num_onode);
            }
        }
    }
    for i_onode in 0..num_onode {
        assert!(onode2nvtx[i_onode] > 0.);
        let s = 1.0 / onode2nvtx[i_onode] as f32;
        onode2gcunit[i_onode][0] *= s;
        onode2gcunit[i_onode][1] *= s;
        onode2gcunit[i_onode][2] *= s;
        let pos_vtx_unit = del_geo_core::mat4_col_major::transform_homogeneous(
            transform_world2unit,
            &onode2gcunit[i_onode],
        )
        .unwrap();
        onode2gcunit[i_onode][0] = pos_vtx_unit[0];
        onode2gcunit[i_onode][1] = pos_vtx_unit[1];
        onode2gcunit[i_onode][2] = pos_vtx_unit[2];
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
        (0..num_vtx * NDIM)
            .map(|_| rng.random::<f32>() * 3. - 1.)
            .collect()
    };
    let transform_world2unit = {
        let m1 = del_geo_core::mat3_col_major::from_translate(&[1., 1.]);
        let m2 = del_geo_core::mat3_col_major::from_diagonal(&[1. / 3., 1. / 3., 1.]);
        del_geo_core::mat3_col_major::mult_mat_col_major(&m2, &m1)
    };
    let (idx2morton, idx2jdx_offset, jdx2vtx) = {
        let mut jdx2vtx = vec![0u32; num_vtx];
        let mut jdx2morton = vec![0u32; num_vtx];
        let mut vtx2morton = vec![0u32; num_vtx];
        crate::mortons::sorted_morten_code2(
            &mut jdx2vtx,
            &mut jdx2morton,
            &mut vtx2morton,
            &vtx2xyz,
            &transform_world2unit,
        );
        crate::mortons::check_morton_code_range_split(&jdx2morton);
        //
        let mut jdx2idx = vec![0u32; num_vtx];
        crate::array1d::unique_for_sorted_array(&jdx2morton, &mut jdx2idx);
        let num_idx = *jdx2idx.last().unwrap() as usize + 1;
        let mut idx2jdx_offset = vec![0u32; num_idx + 1];
        crate::map_idx::inverse(&jdx2idx, &mut idx2jdx_offset);
        let (idx2morton, idx2jdx_offset) = {
            let mut idx2morton = vec![0u32; num_idx];
            for idx in 0..num_idx as usize {
                let jdx = idx2jdx_offset[idx] as usize;
                assert!(jdx < num_vtx as usize);
                idx2morton[idx] = jdx2morton[jdx];
            }
            (idx2morton, idx2jdx_offset)
        };
        (idx2morton, idx2jdx_offset, jdx2vtx)
    };
    {
        // bvh creation
        let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
        let mut bnode2depth = vec![0u32; num_vtx - 1];
        binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
        crate::mortons::check_binary_radix_tree(&bnodes, &idx2morton);
        let mut bnode2onode = vec![0u32; num_vtx - 1];
        let mut idx2bnode = vec![u32::MAX; num_vtx];
        bnode2onode_and_idx2bnode(&bnodes, &bnode2depth, &mut bnode2onode, &mut idx2bnode);
        let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
        // println!("num octree node branch:{}", num_onode);
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
        check_octree_vtx2xyz::<NDIM, { (NDIM + 1) * (NDIM + 1) }>(
            &vtx2xyz,
            &transform_world2unit,
            &idx2jdx_offset,
            &jdx2vtx,
            &idx2onode,
            &idx2center,
            max_depth,
            &onode2center,
            &onode2depth,
        );
    }
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
        (0..num_vtx * NDIM)
            .map(|_| rng.random::<f32>() * 3. - 1.)
            .collect()
    };
    let transform_world2unit = {
        let m1 = del_geo_core::mat4_col_major::from_translate(&[1., 1., 1.]);
        let m2 = del_geo_core::mat4_col_major::from_scale_uniform(1. / 3.);
        del_geo_core::mat4_col_major::mult_mat_col_major(&m2, &m1)
    };
    let (idx2morton, idx2jdx_offset, jdx2vtx) = {
        let mut jdx2vtx = vec![0u32; num_vtx];
        let mut jdx2morton = vec![0u32; num_vtx];
        let mut vtx2morton = vec![0u32; num_vtx];
        crate::mortons::sorted_morten_code3(
            &mut jdx2vtx,
            &mut jdx2morton,
            &mut vtx2morton,
            &vtx2xyz,
            &transform_world2unit,
        );
        crate::mortons::check_morton_code_range_split(&jdx2morton);
        //
        let mut jdx2idx = vec![0u32; num_vtx];
        crate::array1d::unique_for_sorted_array(&jdx2morton, &mut jdx2idx);
        let num_idx = *jdx2idx.last().unwrap() as usize + 1;
        let mut idx2jdx_offset = vec![0u32; num_idx + 1];
        crate::map_idx::inverse(&jdx2idx, &mut idx2jdx_offset);
        let (idx2morton, idx2jdx_offset) = {
            let mut idx2morton = vec![0u32; num_idx];
            for idx in 0..num_idx as usize {
                let jdx = idx2jdx_offset[idx] as usize;
                assert!(jdx < num_vtx as usize);
                idx2morton[idx] = jdx2morton[jdx];
            }
            (idx2morton, idx2jdx_offset)
        };
        (idx2morton, idx2jdx_offset, jdx2vtx)
    };
    let (idx2onode, idx2center, onodes, onode2depth, onode2center) = {
        // bvh creation
        let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
        let mut bnode2depth = vec![0u32; num_vtx - 1];
        binary_radix_tree_and_depth(&idx2morton, NDIM, max_depth, &mut bnodes, &mut bnode2depth);
        crate::mortons::check_binary_radix_tree(&bnodes, &idx2morton);
        let mut bnode2onode = vec![0u32; num_vtx - 1];
        let mut idx2bnode = vec![u32::MAX; num_vtx];
        bnode2onode_and_idx2bnode(&bnodes, &bnode2depth, &mut bnode2onode, &mut idx2bnode);
        let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
        // println!("num octree node branch:{}", num_onode);
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
        (idx2onode, idx2center, onodes, onode2depth, onode2center)
    };
    check_octree::<NDIM>(
        &idx2onode,
        &idx2center,
        &onodes,
        &onode2depth,
        &onode2center,
        max_depth,
    );
    check_octree_vtx2xyz::<3, 16>(
        &vtx2xyz,
        &transform_world2unit,
        &idx2jdx_offset,
        &jdx2vtx,
        &idx2onode,
        &idx2center,
        max_depth,
        &onode2center,
        &onode2depth,
    );
}

#[allow(clippy::type_complexity)]
pub fn construct_octree(
    idx2morton: &[u32],
    max_depth: usize,
) -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
    let num_idx = idx2morton.len();
    let mut bnodes = vec![0u32; (num_idx - 1) * 3];
    let mut bnode2depth = vec![0u32; (num_idx - 1) * 3];
    crate::quad_oct_tree::binary_radix_tree_and_depth(
        idx2morton,
        3,
        max_depth,
        &mut bnodes,
        &mut bnode2depth,
    );
    let mut bnode2onode = vec![0u32; num_idx - 1];
    let mut idx2bnode = vec![0u32; num_idx];
    crate::quad_oct_tree::bnode2onode_and_idx2bnode(
        &bnodes,
        &bnode2depth,
        &mut bnode2onode,
        &mut idx2bnode,
    );
    let num_onode = bnode2onode[num_idx - 2] as usize + 1;
    let mut onodes = vec![u32::MAX; num_onode * 9];
    let mut onode2depth = vec![0u32; num_onode];
    let mut onode2center = vec![0f32; num_onode * 3];
    let mut idx2center = vec![0f32; num_idx * 3];
    let mut idx2onode = vec![0u32; num_idx];
    make_tree_from_binary_radix_tree(
        &bnodes,
        &bnode2onode,
        &bnode2depth,
        &idx2bnode,
        idx2morton,
        num_onode,
        max_depth,
        3,
        &mut onodes,
        &mut onode2depth,
        &mut onode2center,
        &mut idx2onode,
        &mut idx2center,
    );
    check_octree::<3>(
        &idx2onode,
        &idx2center,
        &onodes,
        &onode2depth,
        &onode2center,
        max_depth,
    );
    (onodes, onode2center, onode2depth, idx2onode, idx2center)
}
