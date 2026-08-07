use num_traits::AsPrimitive;

fn dominant_direction_pca<T>(remaining_elems: &[usize], elem2center: &[[T; 3]]) -> ([T; 3], [T; 3])
where
    T: num_traits::Float + 'static + Copy,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let mut org = [T::zero(); 3];
    for &i_tri in remaining_elems {
        // center of the gravity of list
        org.add_in_place(&elem2center[i_tri]);
    }
    org.scale_in_place(T::one() / remaining_elems.len().as_());
    let mut cov = [T::zero(); 9];
    for &i_tri in remaining_elems {
        let v = elem2center[i_tri].sub(&org);
        let cov0 = &del_geo_core::mat3_col_major::from_scaled_outer_product(T::one(), &v, &v);
        cov = del_geo_core::mat3_col_major::add(&cov, cov0);
    }
    let mut dir = [T::one(), T::one(), T::one()];
    for _ in 0..10 {
        // power method to find the max eigen value/vector
        dir = del_geo_core::mat3_col_major::mult_vec(&cov, &dir);
        dir = dir.normalize();
    }
    (org, dir)
}

#[allow(dead_code)]
fn dominant_direction_aabb(remaining_elems: &[usize], elem2center: &[f32]) -> ([f32; 3], [f32; 3]) {
    let aabb =
        crate::vtx2xyz::aabb3_indexed(remaining_elems, elem2center.as_chunks::<3>().0, 1.0e-6);
    let lenx = aabb[3] - aabb[0];
    let leny = aabb[4] - aabb[1];
    let lenz = aabb[5] - aabb[2];
    let mut dir = [0f32; 3]; // longest direction of AABB
    if lenx > leny && lenx > lenz {
        dir[0] = 1.;
    }
    if leny > lenz && leny > lenx {
        dir[1] = 1.;
    }
    if lenz > lenx && lenz > leny {
        dir[2] = 1.;
    }
    let org = [
        (aabb[0] + aabb[3]) * 0.5,
        (aabb[1] + aabb[4]) * 0.5,
        (aabb[2] + aabb[5]) * 0.5,
    ];
    (org, dir)
}

fn divide_list_of_elements<T>(
    i_node_root: usize,
    elem2bvhnode: &mut [usize],
    bvhnodes: &mut Vec<[usize; 3]>,
    remaining_elems: &[usize],
    num_adjacent_elems: usize,
    elem2elem: &[usize],
    elem2center: &[[T; 3]],
) where
    T: num_traits::Float + Copy + 'static,
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let inode_ch0 = bvhnodes.len();
    let inode_ch1 = inode_ch0 + 1;
    bvhnodes.resize(bvhnodes.len() + 2, [usize::MAX; 3]);
    bvhnodes[inode_ch0][0] = i_node_root;
    bvhnodes[inode_ch1][0] = i_node_root;
    bvhnodes[i_node_root][1] = inode_ch0;
    bvhnodes[i_node_root][2] = inode_ch1;
    let list_ch0 = {
        let mut list_ch0 = vec![0_usize; 0];
        if remaining_elems.len() == 2 {
            let i_tri0 = remaining_elems[0];
            list_ch0.push(i_tri0);
            elem2bvhnode[i_tri0] = inode_ch0;
        } else {
            // extract the triangles in the child node 0
            assert!(remaining_elems.len() > 1);
            let (org, mut dir) = dominant_direction_pca(remaining_elems, elem2center);
            let i_elem_ker = {
                // pick one element
                let mut i_elem_ker = usize::MAX;
                for &i_elem in remaining_elems {
                    let cntr = &elem2center[i_elem];
                    let det0 = cntr.sub(&org).dot(&dir);
                    if det0.abs() < 1.0e-10f64.as_() {
                        continue;
                    }
                    if det0 < T::zero() {
                        dir = dir.scale(-T::one());
                    }
                    i_elem_ker = i_elem;
                    break;
                }
                i_elem_ker
            };
            // triangles connected to `itri_ker` and in the direction of `dir`
            elem2bvhnode[i_elem_ker] = inode_ch0;
            list_ch0.push(i_elem_ker);
            let mut elem_stack = vec![0_usize; 0];
            elem_stack.push(i_elem_ker);
            while let Some(itri0) = elem_stack.pop() {
                for i_face in 0..num_adjacent_elems {
                    let j_elem = elem2elem[itri0 * num_adjacent_elems + i_face];
                    if j_elem == usize::MAX {
                        continue;
                    }
                    if elem2bvhnode[j_elem] != i_node_root {
                        continue;
                    }
                    let cntr = &elem2center[j_elem];
                    if cntr.sub(&org).dot(&dir) < T::zero() {
                        continue;
                    }
                    elem_stack.push(j_elem);
                    elem2bvhnode[j_elem] = inode_ch0;
                    list_ch0.push(j_elem);
                }
            }
        }
        list_ch0
    };
    assert!(!list_ch0.is_empty());
    // extract the triangles in child node １
    // exclude the triangles that is included in the child node 0
    let mut list_ch1 = vec![0_usize; 0];
    for &i_tri in remaining_elems {
        if elem2bvhnode[i_tri] == inode_ch0 {
            continue;
        }
        assert_eq!(elem2bvhnode[i_tri], i_node_root);
        elem2bvhnode[i_tri] = inode_ch1;
        list_ch1.push(i_tri);
    }
    assert!(!list_ch1.is_empty());
    // ---------------------------
    if list_ch0.len() == 1 {
        bvhnodes[inode_ch0][1] = list_ch0[0];
        bvhnodes[inode_ch0][2] = usize::MAX;
    } else {
        // subdivide child node 0
        divide_list_of_elements(
            inode_ch0,
            elem2bvhnode,
            bvhnodes,
            &list_ch0,
            num_adjacent_elems,
            elem2elem,
            elem2center,
        );
    }
    // -----------------------------
    if list_ch1.len() == 1 {
        bvhnodes[inode_ch1][1] = list_ch1[0];
        bvhnodes[inode_ch1][2] = usize::MAX;
    } else {
        // subdivide the child node 1
        divide_list_of_elements(
            inode_ch1,
            elem2bvhnode,
            bvhnodes,
            &list_ch1,
            num_adjacent_elems,
            elem2elem,
            elem2center,
        );
    }
}

pub fn from_uniform_mesh_with_elem2elem_elem2center<T>(
    elem2elem: &[usize],
    num_adjacent_elems: usize,
    elem2center: &[[T; 3]],
) -> Vec<[usize; 3]>
where
    T: num_traits::Float + Copy + 'static,
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let nelem = elem2center.len();
    let remaining_elems: Vec<usize> = (0..nelem).collect();
    let mut elem2node = vec![0; nelem];
    let mut nodes = vec![[usize::MAX; 3]; 1];
    divide_list_of_elements(
        0,
        &mut elem2node,
        &mut nodes,
        &remaining_elems,
        num_adjacent_elems,
        elem2elem,
        elem2center,
    );
    nodes
}

pub fn from_triangle_mesh<T>(tri2vtx: &[[usize; 3]], vtx2xyz: &[[T; 3]]) -> Vec<[usize; 3]>
where
    T: num_traits::Float + std::ops::AddAssign + 'static + Copy,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let tri2tri = crate::elem2elem::from_uniform_mesh::<usize, 3>(
        tri2vtx,
        &del_geo_core::tri::FACE2IDX,
        &del_geo_core::tri::IDX2NODE,
        vtx2xyz.len(),
    );
    let tri2center =
        crate::elem2center::from_uniform_mesh_as_points::<_, _, 3, 3>(tri2vtx, vtx2xyz);
    from_uniform_mesh_with_elem2elem_elem2center(&tri2tri, 3, &tri2center)
}
