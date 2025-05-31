//! methods related to triangle mesh topology

/// split polygons of polygonal mesh into triangles
pub fn from_polygon_mesh(elem2idx: &[usize], idx2vtx: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut num_tri = 0_usize;
    for i_elem in 0..elem2idx.len() - 1 {
        assert!(elem2idx[i_elem + 1] >= elem2idx[i_elem]);
        let num_node = elem2idx[i_elem + 1] - elem2idx[i_elem];
        num_tri += num_node - 2;
    }
    let mut tri2vtx = Vec::<usize>::with_capacity(num_tri * 3);
    let mut new2old = Vec::<usize>::with_capacity(num_tri);
    for i_elem in 0..elem2idx.len() - 1 {
        let num_node = elem2idx[i_elem + 1] - elem2idx[i_elem];
        let idx0 = elem2idx[i_elem];
        for i_node in 0..num_node - 2 {
            tri2vtx.push(idx2vtx[idx0]);
            tri2vtx.push(idx2vtx[idx0 + 1 + i_node]);
            tri2vtx.push(idx2vtx[idx0 + 2 + i_node]);
            new2old.push(i_elem);
        }
    }
    (tri2vtx, new2old)
}

/// split quad element to triangle element
pub fn from_quad_mesh(quad2vtx: &[usize]) -> Vec<usize> {
    let nquad = quad2vtx.len() / 4;
    let mut tri2vtx = vec![0; nquad * 2 * 3];
    for iquad in 0..nquad {
        tri2vtx[iquad * 6] = quad2vtx[iquad * 4];
        tri2vtx[iquad * 6 + 1] = quad2vtx[iquad * 4 + 1];
        tri2vtx[iquad * 6 + 2] = quad2vtx[iquad * 4 + 2];
        //
        tri2vtx[iquad * 6 + 3] = quad2vtx[iquad * 4];
        tri2vtx[iquad * 6 + 4] = quad2vtx[iquad * 4 + 2];
        tri2vtx[iquad * 6 + 5] = quad2vtx[iquad * 4 + 3];
    }
    tri2vtx
}

pub fn from_grid(nx: usize, ny: usize) -> Vec<usize> {
    let mx = nx - 1;
    let my = ny - 1;
    let ncell = mx * my;
    let mut tri2vtx = Vec::<usize>::with_capacity(ncell * 2 * 3);
    for imy in 0..my {
        for imx in 0..mx {
            let ivx0 = imx + imy * nx;
            let ivx1 = imx + 1 + imy * nx;
            let ivx2 = imx + 1 + (imy + 1) * nx;
            let ivx3 = imx + (imy + 1) * nx;
            tri2vtx.extend([ivx0, ivx1, ivx2]);
            tri2vtx.extend([ivx0, ivx2, ivx3]);
        }
    }
    assert_eq!(tri2vtx.len(), ncell * 2 * 3);
    tri2vtx
}

// above: from methods
// ---------------------------------

/// find node index of a triangle from the vertex index
pub fn find_node_tri(tri2vtx: &[usize], i_vtx: usize) -> usize {
    if tri2vtx[0] == i_vtx {
        return 0;
    }
    if tri2vtx[1] == i_vtx {
        return 1;
    }
    if tri2vtx[2] == i_vtx {
        return 2;
    }
    panic!();
}

/// triangle surrounding triangle
pub fn elem2elem(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize,
    is_simplex: bool,
) -> Vec<usize> {
    let (face2idx, idx2node) = if is_simplex {
        crate::elem2elem::face2node_of_simplex_element(num_node)
    } else {
        crate::elem2elem::face2node_of_polygon_element(num_node)
    };
    crate::elem2elem::from_uniform_mesh(elem2vtx, num_node, &face2idx, &idx2node, num_vtx)
}
