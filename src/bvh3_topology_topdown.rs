use num_traits::AsPrimitive;
fn dominant_direction_pca<T>(
    remaining_elems: &[usize],
    elem2center: &[T],
) -> (nalgebra::Vector3<T>, nalgebra::Vector3<T>)
where
    T: nalgebra::RealField + 'static + Copy,
    usize: AsPrimitive<T>,
{
    let mut org = nalgebra::Vector3::<T>::zeros();
    for i_tri in remaining_elems {
        // center of the gravity of list
        org += nalgebra::Vector3::<T>::from_row_slice(&elem2center[i_tri * 3..i_tri * 3 + 3]);
    }
    org /= remaining_elems.len().as_();
    let mut cov = nalgebra::Matrix3::<T>::zeros();
    for i_tri in remaining_elems {
        let v =
            nalgebra::Vector3::<T>::from_row_slice(&elem2center[i_tri * 3..i_tri * 3 + 3]) - org;
        cov += v * v.transpose();
    }
    let mut dir = nalgebra::Vector3::<T>::new(T::one(), T::one(), T::one());
    for _ in 0..10 {
        // power method to find the max eigen value/vector
        dir = cov * dir;
        dir = dir.normalize();
    }
    (org, dir)
}

#[allow(dead_code)]
fn dominant_direction_aabb(
    remaining_elems: &[usize],
    elem2center: &[f32],
) -> (nalgebra::Vector3<f32>, nalgebra::Vector3<f32>) {
    let aabb = del_geo::aabb3::from_list_of_vertices(remaining_elems, elem2center, 1.0e-6);
    let lenx = aabb[3] - aabb[0];
    let leny = aabb[4] - aabb[1];
    let lenz = aabb[5] - aabb[2];
    let mut dir = nalgebra::Vector3::<f32>::zeros(); // longest direction of AABB
    if lenx > leny && lenx > lenz {
        dir.x = 1.;
    }
    if leny > lenz && leny > lenx {
        dir.y = 1.;
    }
    if lenz > lenx && lenz > leny {
        dir.z = 1.;
    }
    let org = nalgebra::Vector3::<f32>::new(
        (aabb[0] + aabb[3]) * 0.5,
        (aabb[1] + aabb[4]) * 0.5,
        (aabb[2] + aabb[5]) * 0.5,
    );
    (org, dir)
}

#[allow(clippy::identity_op)]
fn divide_list_of_elements<T>(
    i_node_root: usize,
    elem2node: &mut [usize],
    nodes: &mut Vec<usize>,
    remaining_elems: &[usize],
    num_adjacent_elems: usize,
    elem2elem: &[usize],
    elem2center: &[T],
) where
    T: nalgebra::RealField + Copy + 'static,
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let inode_ch0 = nodes.len() / 3;
    let inode_ch1 = inode_ch0 + 1;
    nodes.resize(nodes.len() + 6, usize::MAX);
    nodes[inode_ch0 * 3 + 0] = i_node_root;
    nodes[inode_ch1 * 3 + 0] = i_node_root;
    nodes[i_node_root * 3 + 1] = inode_ch0;
    nodes[i_node_root * 3 + 2] = inode_ch1;
    let list_ch0 = {
        let mut list_ch0 = vec![0_usize; 0];
        if remaining_elems.len() == 2 {
            let i_tri0 = remaining_elems[0];
            list_ch0.push(i_tri0);
            elem2node[i_tri0] = inode_ch0;
        } else {
            // extract the triangles in the child node 0
            assert!(remaining_elems.len() > 1);
            let (org, mut dir) = dominant_direction_pca(remaining_elems, elem2center);
            let i_elem_ker = {
                // pick one element
                let mut i_elem_ker = usize::MAX;
                for &i_elem in remaining_elems {
                    let cntr = nalgebra::Vector3::<T>::from_row_slice(
                        &elem2center[i_elem * 3..i_elem * 3 + 3],
                    );
                    let det0 = (cntr - org).dot(&dir);
                    if det0.abs() < 1.0e-10f64.as_() {
                        continue;
                    }
                    if det0 < T::zero() {
                        dir *= -(T::one());
                    }
                    i_elem_ker = i_elem;
                    break;
                }
                i_elem_ker
            };
            // triangles connected to `itri_ker` and in the direction of `dir`
            elem2node[i_elem_ker] = inode_ch0;
            list_ch0.push(i_elem_ker);
            let mut elem_stack = vec![0_usize; 0];
            elem_stack.push(i_elem_ker);
            while let Some(itri0) = elem_stack.pop() {
                for i_face in 0..num_adjacent_elems {
                    let j_elem = elem2elem[itri0 * num_adjacent_elems + i_face];
                    if j_elem == usize::MAX {
                        continue;
                    }
                    if elem2node[j_elem] != i_node_root {
                        continue;
                    }
                    let cntr = nalgebra::Vector3::<T>::from_row_slice(
                        &elem2center[j_elem * 3..j_elem * 3 + 3],
                    );
                    if (cntr - org).dot(&dir) < T::zero() {
                        continue;
                    }
                    elem_stack.push(j_elem);
                    elem2node[j_elem] = inode_ch0;
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
        if elem2node[i_tri] == inode_ch0 {
            continue;
        }
        assert_eq!(elem2node[i_tri], i_node_root);
        elem2node[i_tri] = inode_ch1;
        list_ch1.push(i_tri);
    }
    assert!(!list_ch1.is_empty());
    // ---------------------------
    if list_ch0.len() == 1 {
        nodes[inode_ch0 * 3 + 1] = list_ch0[0];
        nodes[inode_ch0 * 3 + 2] = usize::MAX;
    } else {
        // subdivide child node 0
        divide_list_of_elements(
            inode_ch0,
            elem2node,
            nodes,
            &list_ch0,
            num_adjacent_elems,
            elem2elem,
            elem2center,
        );
    }
    // -----------------------------
    if list_ch1.len() == 1 {
        nodes[inode_ch1 * 3 + 1] = list_ch1[0];
        nodes[inode_ch1 * 3 + 2] = usize::MAX;
    } else {
        // subdivide the child node 1
        divide_list_of_elements(
            inode_ch1,
            elem2node,
            nodes,
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
    elem2center: &[T],
) -> Vec<usize>
where
    T: nalgebra::RealField + Copy + 'static,
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let nelem = elem2center.len() / 3;
    let remaining_elems: Vec<usize> = (0..nelem).collect();
    let mut elem2node = vec![0; nelem];
    let mut nodes = vec![usize::MAX; 3];
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

pub fn from_triangle_mesh<T>(tri2vtx: &[usize], vtx2xyz: &[T]) -> Vec<usize>
where
    T: num_traits::Float + std::ops::AddAssign + 'static + Copy + nalgebra::RealField,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let (face2idx, idx2node) = crate::elem2elem::face2node_of_simplex_element(3);
    let tri2tri =
        crate::elem2elem::from_uniform_mesh(tri2vtx, 3, &face2idx, &idx2node, vtx2xyz.len() / 3);
    let tri2center = crate::elem2center::from_uniform_mesh_as_points(tri2vtx, 3, vtx2xyz, 3);
    from_uniform_mesh_with_elem2elem_elem2center(&tri2tri, 3, &tri2center)
}
