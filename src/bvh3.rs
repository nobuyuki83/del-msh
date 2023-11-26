fn dominant_direction_pca(
    remaining_elems: &[usize],
    elem2center: &[f32]) -> (nalgebra::Vector3::<f32>, nalgebra::Vector3::<f32>)
{
    let mut org = nalgebra::Vector3::<f32>::zeros();
    for i_tri in remaining_elems { // center of the gravity of list
        org += nalgebra::Vector3::<f32>::from_row_slice(&elem2center[i_tri * 3..i_tri * 3 + 3]);
    }
    org /= remaining_elems.len() as f32;
    let mut cov = nalgebra::Matrix3::<f32>::zeros();
    for i_tri in remaining_elems {
        let v = nalgebra::Vector3::<f32>::from_row_slice(
            &elem2center[i_tri * 3..i_tri * 3 + 3]) - org;
        cov += v * v.transpose();
    }
    let mut dir = nalgebra::Vector3::<f32>::new(1., 1., 1.);
    for _ in 0..10 {// power method to find the max eigen value/vector
        dir = cov * dir;
        dir = dir.normalize();
    }
    (org, dir)
}

#[allow(dead_code)]
fn dominant_direction_aabb(
    remaining_elems: &[usize],
    elem2center: &[f32]) -> (nalgebra::Vector3::<f32>, nalgebra::Vector3::<f32>)
{
    let aabb = del_geo::aabb3::from_list_of_vertices(
        remaining_elems, elem2center, 1.0e-6);
    let lenx = aabb.1.x - aabb.0.x;
    let leny = aabb.1.y - aabb.0.y;
    let lenz = aabb.1.z - aabb.0.z;
    let mut dir = nalgebra::Vector3::<f32>::zeros(); // longest direction of AABB
    if lenx > leny && lenx > lenz { dir.x = 1.; }
    if leny > lenz && leny > lenx { dir.y = 1.; }
    if lenz > lenx && lenz > leny { dir.z = 1.; }
    let org = nalgebra::Vector3::<f32>::new(
        (aabb.0.x + aabb.1.x) * 0.5,
        (aabb.0.y + aabb.1.y) * 0.5,
        (aabb.0.z + aabb.1.z) * 0.5);
    (org, dir)
}

#[allow(clippy::identity_op)]
fn divide_list_of_elements(
    i_node_root: usize,
    elem2node: &mut [usize],
    nodes: &mut Vec<usize>,
    remaining_elems: &[usize],
    num_adjacent_elems: usize,
    elem2elem: &[usize],
    elem2center: &[f32])
{
    assert!(remaining_elems.len() > 1);
    let (org, mut dir) = dominant_direction_pca(
        remaining_elems, elem2center);
    let i_elem_ker = { // pick one element
        let mut i_elem_ker = usize::MAX;
        for &i_elem in remaining_elems {
            let cntr = nalgebra::Vector3::<f32>::from_row_slice(
                &elem2center[i_elem * 3..i_elem * 3 + 3]);
            let det0 = (cntr - org).dot(&dir);
            if det0.abs() < 1.0e-10 { continue; }
            if det0 < 0. { dir *= -1.; }
            i_elem_ker = i_elem;
            break;
        }
        i_elem_ker
    };
    let inode_ch0 = nodes.len() / 3;
    let inode_ch1 = inode_ch0 + 1;
    nodes.resize(nodes.len() + 6, usize::MAX);
    nodes[inode_ch0 * 3 + 0] = i_node_root;
    nodes[inode_ch1 * 3 + 0] = i_node_root;
    nodes[i_node_root * 3 + 1] = inode_ch0;
    nodes[i_node_root * 3 + 2] = inode_ch1;
    let mut list_ch0 = vec!(0_usize; 0);
    {
        // extract the triangles in the child node 0
        // triangles connected to `itri_ker` and in the direction of `dir`
        elem2node[i_elem_ker] = inode_ch0;
        list_ch0.push(i_elem_ker);
        let mut elem_stack = vec!(0_usize; 0);
        elem_stack.push(i_elem_ker);
        while let Some(itri0) = elem_stack.pop() {
            for i_face in 0..num_adjacent_elems {
                let j_elem = elem2elem[itri0 * num_adjacent_elems + i_face];
                if j_elem == usize::MAX { continue; }
                if elem2node[j_elem] != i_node_root { continue; }
                let cntr = nalgebra::Vector3::<f32>::from_row_slice(
                    &elem2center[j_elem * 3..j_elem * 3 + 3]);
                if (cntr - org).dot(&dir) < 0. { continue; }
                elem_stack.push(j_elem);
                elem2node[j_elem] = inode_ch0;
                list_ch0.push(j_elem);
            }
        }
        assert!(!list_ch0.is_empty());
    }
    // extract the triangles in child node １
    // exclude the triangles that is included in the child node 0
    let mut list_ch1 = vec!(0_usize; 0);
    for &i_tri in remaining_elems {
        if elem2node[i_tri] == inode_ch0 { continue; }
        assert_eq!(elem2node[i_tri], i_node_root);
        elem2node[i_tri] = inode_ch1;
        list_ch1.push(i_tri);
    }
    assert!(!list_ch1.is_empty());
// ---------------------------
    if list_ch0.len() == 1 {
        nodes[inode_ch0 * 3 + 1] = list_ch0[0];
        nodes[inode_ch0 * 3 + 2] = usize::MAX;
    } else { // subdivide child node 0
        divide_list_of_elements(
            inode_ch0, elem2node, nodes,
            &list_ch0, num_adjacent_elems, elem2elem, elem2center);
    }
    list_ch0.clear();
    // -----------------------------
    if list_ch1.len() == 1 {
        nodes[inode_ch1 * 3 + 1] = list_ch1[0];
        nodes[inode_ch1 * 3 + 2] = usize::MAX;
    } else { // subdivide the child node 1
        divide_list_of_elements(
            inode_ch1, elem2node, nodes,
            &list_ch1, num_adjacent_elems, elem2elem, elem2center);
    }
}


pub fn build_topology_for_uniform_mesh_with_elem2elem_elem2center(
    elem2elem: &[usize],
    num_adjacent_elems: usize,
    elem2center: &[f32]) -> Vec<usize>
{
    let nelem = elem2center.len() / 3;
    let remaining_elems: Vec<usize> = (0..nelem).collect();
    let mut elem2node = vec!(0; nelem);
    let mut nodes = vec!(usize::MAX; 3);
    divide_list_of_elements(
        0, &mut elem2node, &mut nodes,
        &remaining_elems, num_adjacent_elems, elem2elem, elem2center);
    nodes
}



#[allow(clippy::identity_op)]
pub fn build_geometry_aabb_for_uniform_mesh(
    aabbs: &mut [f32],
    i_bvhnode: usize,
    bvhnodes: &[usize],
    elem2vtx: &[usize],
    num_noel: usize,
    vtx2xyz: &[f32])
{
    // aabbs.resize();
    assert_eq!(aabbs.len() / 6, bvhnodes.len() / 3);
    assert!(i_bvhnode < bvhnodes.len() / 3);
    let i_bvhnode_child0 = bvhnodes[i_bvhnode * 3 + 1];
    let i_bvhnode_child1 = bvhnodes[i_bvhnode * 3 + 2];
    if i_bvhnode_child1 == usize::MAX { // leaf node
        let i_elem = i_bvhnode_child0;
        let aabb = del_geo::aabb3::from_list_of_vertices(
            &elem2vtx[i_elem * num_noel..(i_elem + 1) * num_noel],
            vtx2xyz, 0.0);
        aabbs[i_bvhnode * 6 + 0..i_bvhnode * 6 + 3].copy_from_slice(aabb.0.as_slice());
        aabbs[i_bvhnode * 6 + 3..i_bvhnode * 6 + 6].copy_from_slice(aabb.1.as_slice());
    } else {  // branch node
        assert_eq!(bvhnodes[i_bvhnode_child0 * 3 + 0], i_bvhnode);
        assert_eq!(bvhnodes[i_bvhnode_child1 * 3 + 0], i_bvhnode);
        // build right tree
        build_geometry_aabb_for_uniform_mesh(
            aabbs,
            i_bvhnode_child0, bvhnodes, elem2vtx, num_noel, vtx2xyz);
        // build left tree
        build_geometry_aabb_for_uniform_mesh(
            aabbs,
            i_bvhnode_child1, bvhnodes, elem2vtx, num_noel, vtx2xyz);
        let aabb = del_geo::aabb3::from_two_aabbs_slice6(
            &aabbs[i_bvhnode_child0 * 6..(i_bvhnode_child0 + 1) * 6],
            &aabbs[i_bvhnode_child1 * 6..(i_bvhnode_child1 + 1) * 6]);
        aabbs[i_bvhnode * 6..(i_bvhnode + 1) * 6].copy_from_slice(&aabb);
    }
}