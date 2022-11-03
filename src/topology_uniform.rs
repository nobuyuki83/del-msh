//! utility library to find topology information of uniform mesh
//! uniform mesh is a mesh that has single type of element (quad, tri, tet)

/// element surrounding points
pub fn elsup(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize) -> (Vec<usize>, Vec<usize>) {
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2vtx.len(), num_elem* num_node);
    let mut vtx2elem_idx= vec!(0_usize; num_vtx + 1);
    for ielem in 0..num_elem {
        for inoel in 0..num_node {
            let ino1 = elem2vtx[ielem * num_node + inoel];
            let ino1 = ino1 as usize;
            vtx2elem_idx[ino1 + 1] += 1;
        }
    }
    for ivtx in 0..num_vtx {
        vtx2elem_idx[ivtx + 1] += vtx2elem_idx[ivtx];
    }
    let nelsup = vtx2elem_idx[num_vtx];
    let mut vtx2elem = Vec::<usize>::new();
    vtx2elem.resize(nelsup, 0);
    for ielem in 0..num_elem {
        for inode in 0..num_node {
            let ivtx1 = elem2vtx[ielem * num_node + inode];
            let ivtx1 = ivtx1 as usize;
            let ielem0 = vtx2elem_idx[ivtx1];
            vtx2elem[ielem0] = ielem;
            vtx2elem_idx[ivtx1] += 1;
        }
    }
    for ivtx in (1..num_vtx).rev() {
        vtx2elem_idx[ivtx] = vtx2elem_idx[ivtx - 1];
    }
    vtx2elem_idx[0] = 0;
    (vtx2elem_idx, vtx2elem)
}

/// point surrounding point for mesh
/// * elem_vtx - map element to vertex: list of vertex index for each element
/// * vtx_elem_idx - map vertex to element index: cumulative sum
/// * vtx_elem - map vertex to element value: list of value
/// * num_vtx_par_elem - number of vertex par elemnent (e.g., 3 for tri, 4 for quad)
/// * num_vtx - number of vertex
pub fn psup(
    elem2vtx: &[usize],
    vtx2elem_idx: &[usize],
    vtx2elem: &[usize],
    num_vtx_par_elem: usize,
    num_vtx: usize) -> (Vec<usize>, Vec<usize>) {
    let mut vtx2vtx_idx = Vec::<usize>::new();
    let mut vtx2flg = Vec::<usize>::new();
    vtx2flg.resize(num_vtx, usize::MAX);
    vtx2vtx_idx.resize(num_vtx + 1, Default::default());
    for i_vtx in 0..num_vtx {
        vtx2flg[i_vtx] = i_vtx;
        for ielsup in vtx2elem_idx[i_vtx]..vtx2elem_idx[i_vtx + 1] {
            let jelem = vtx2elem[ielsup];
            for jnoel in 0..num_vtx_par_elem {
                let jnode = elem2vtx[jelem * num_vtx_par_elem + jnoel];
                if vtx2flg[jnode] != i_vtx {
                    vtx2flg[jnode] = i_vtx;
                    vtx2vtx_idx[i_vtx + 1] += 1;
                }
            }
        }
    }
    for ipoint in 0..num_vtx {
        vtx2vtx_idx[ipoint + 1] += vtx2vtx_idx[ipoint];
    }
    let npsup = vtx2vtx_idx[num_vtx];
    let mut vtx2vtx= vec!(0_usize; npsup);
    for i_vtx in 0..num_vtx { vtx2flg[i_vtx] = usize::MAX; }
    for i_vtx in 0..num_vtx {
        vtx2flg[i_vtx] = i_vtx;
        for ielsup in vtx2elem_idx[i_vtx]..vtx2elem_idx[i_vtx + 1] {
            let jelem = vtx2elem[ielsup];
            for jnoel in 0..num_vtx_par_elem {
                let jnode = elem2vtx[jelem * num_vtx_par_elem + jnoel];
                if vtx2flg[jnode] != i_vtx {
                    vtx2flg[jnode] = i_vtx;
                    let ind = vtx2vtx_idx[i_vtx];
                    vtx2vtx[ind] = jnode;
                    vtx2vtx_idx[i_vtx] += 1;
                }
            }
        }
    }
    for i_vtx in (1..num_vtx).rev() {
        vtx2vtx_idx[i_vtx] = vtx2vtx_idx[i_vtx - 1];
    }
    vtx2vtx_idx[0] = 0;
    (vtx2vtx_idx, vtx2vtx)
}


/// element surrounding element
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `vtx2elem_idx` - jagged array index of element surrounding point
/// * `vtx2elem` - jagged array value of  element surrounding point
///
///  triangle: face_node_idx = \[0,2,4,6]; face_node = \[1,2,2,0,0,1];
pub fn elsuel(
    elem2vtx: &[usize],
    num_node: usize,
    vtx2elem_idx: &[usize],
    vtx2elem: &[usize],
    face2node_idx: &[usize],
    face2node: &[usize]) -> Vec<usize> {
    assert!(!vtx2elem_idx.is_empty());
    let num_vtx = vtx2elem_idx.len() - 1;
    let num_face_par_elem = face2node_idx.len() - 1;
    let num_max_node_on_face = {
        let mut n0 = 0_usize;
        for i_face in 0..num_face_par_elem {
            let nno = face2node_idx[i_face + 1] - face2node_idx[i_face];
            n0 = if nno > n0 { nno } else { n0 }
        }
        n0
    };

    let num_elem = elem2vtx.len() / num_node;
    let mut elem2elem = vec!(usize::MAX; num_elem * num_face_par_elem);

    let mut vtx2flag = vec!(0; num_vtx); // vertex index -> flag
    let mut fano2vtx = vec!(0; num_max_node_on_face);  // face node index -> vertex index
    for i_elem in 0..num_elem {
        for i_face in 0..num_face_par_elem {
            for ifano in 0..face2node_idx[i_face + 1]- face2node_idx[i_face] {
                let i_node0 = face2node[ifano+ face2node_idx[i_face]];
                assert!(i_node0 < num_node);
                let i_vtx = elem2vtx[i_elem * num_node + i_node0];
                assert!(i_vtx < num_vtx);
                fano2vtx[ifano] = i_vtx;
                vtx2flag[i_vtx] = 1;
            }
            let i_vtx0 = fano2vtx[0];
            let mut flag0 = false;
            for i_vtx2elem in vtx2elem_idx[i_vtx0]..vtx2elem_idx[i_vtx0 + 1] {
                let j_elem0 = vtx2elem[i_vtx2elem];
                if j_elem0 == i_elem {
                    continue;
                }
                for j_face in 0..num_face_par_elem {
                    flag0 = true;
                    for j_fano in face2node_idx[j_face]..face2node_idx[j_face + 1] {
                        let j_node0 = face2node[j_fano];
                        let j_vtx0 = elem2vtx[j_elem0 * num_node + j_node0];
                        if vtx2flag[j_vtx0] == 0 {
                            flag0 = false;
                            break;
                        }
                    }
                    if flag0 {
                        elem2elem[i_elem * num_face_par_elem + i_face] = j_elem0;
                        break;
                    }
                }
                if flag0 {
                    break;
                }
            }
            if !flag0 {
                elem2elem[i_elem * num_face_par_elem + i_face] = usize::MAX;
            }
            for ifano in 0..face2node_idx[i_face + 1] - face2node_idx[i_face] {
                vtx2flag[fano2vtx[ifano]] = 0;
            }
        }
    }
    elem2elem
}

/// element surrounding element
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `num_vtx` - number of vertices
///
///  triangle: face_node_idx = \[0,2,4,6]; face_node = \[1,2,2,0,0,1];
pub fn elsuel2(
    elem2vtx: &[usize],
    num_node: usize,
    face2node_idx: &[usize],
    face2node: &[usize],
    num_vtx: usize) -> Vec<usize>{
    let vtx2elem = elsup(
        &elem2vtx, num_node,
        num_vtx);
    elsuel(
        &elem2vtx, num_node,
        &vtx2elem.0, &vtx2elem.1,
        face2node_idx, face2node)
}

// ------------------------------

pub fn psup_elem_edge(
    elem2vtx: &[usize],
    num_node_par_elem: usize,
    edges_par_elem: &[usize],
    vtx2elem_idx: &Vec<usize>,
    vtx2elem: &Vec<usize>,
    is_bidirectional: bool) -> (Vec<usize>, Vec<usize>) {
    let num_edge_par_elem = edges_par_elem.len() / 2;
    assert_eq!(edges_par_elem.len(), num_edge_par_elem * 2);
    let mut vtx2vtx = Vec::<usize>::new();

    let num_vtx = vtx2elem_idx.len() - 1;
    let mut vtx2vtx_idx = vec!(0_usize; num_vtx + 1);
    vtx2vtx_idx[0] = 0;
    for i_vtx in 0..num_vtx {
        let mut set_vtx_idx = std::collections::BTreeSet::new();
        for ielsup in vtx2elem_idx[i_vtx]..vtx2elem_idx[i_vtx + 1] {
            let ielem0 = vtx2elem[ielsup];
            for iedge in 0..num_edge_par_elem {
                let inode0 = edges_par_elem[iedge * 2 + 0];
                let inode1 = edges_par_elem[iedge * 2 + 1];
                let ivtx0 = elem2vtx[ielem0 * num_node_par_elem + inode0];
                let ivtx1 = elem2vtx[ielem0 * num_node_par_elem + inode1];
                if ivtx0 != i_vtx && ivtx1 != i_vtx { continue; }
                if ivtx0 == i_vtx {
                    if is_bidirectional || ivtx1 > i_vtx {
                        set_vtx_idx.insert(ivtx1);
                    }
                } else {
                    if is_bidirectional || ivtx0 > i_vtx {
                        set_vtx_idx.insert(ivtx0);
                    }
                }
            }
        }
        for itr in &set_vtx_idx {
            vtx2vtx.push((*itr).try_into().unwrap());
        }
        vtx2vtx_idx[i_vtx + 1] = vtx2vtx_idx[i_vtx] + set_vtx_idx.len();
    }
    (vtx2vtx_idx, vtx2vtx)
}

/// making vertex indexes list of edges from psup (point surrounding point)
pub fn mshline_psup(
    vtx2vtx_idx: &Vec<usize>,
    vtx2vtx: &Vec<usize>) -> Vec<usize> {
    let mut line_vtx = Vec::<usize>::with_capacity(vtx2vtx.len() * 2);
    let np = vtx2vtx_idx.len() - 1;
    for ip in 0..np {
        for ipsup in vtx2vtx_idx[ip]..vtx2vtx_idx[ip + 1] {
            let jp = vtx2vtx[ipsup];
            line_vtx.push(ip);
            line_vtx.push(jp);
        }
    }
    line_vtx
}

pub fn mshline(
    elem2vtx: &[usize],
    num_vtx_par_elem: usize,
    edges_par_elem: &[usize],
    num_vtx: usize) -> Vec<usize>
{
    let vtx2elem = elsup(
        elem2vtx, num_vtx_par_elem, num_vtx);
    let vtx2vtx = psup_elem_edge(
        elem2vtx, num_vtx_par_elem,
        edges_par_elem,
        &vtx2elem.0, &vtx2elem.1,
        false);
    mshline_psup(&vtx2vtx.0, &vtx2vtx.1)
}

pub fn tri_from_quad(
    quad2vtx: &[usize]) -> Vec<usize>
{
    let nquad = quad2vtx.len() / 4;
    let mut tri2vtx = vec![0; nquad*2*3];
    for iquad in 0..nquad {
        tri2vtx[iquad * 6 + 0] = quad2vtx[iquad * 4 + 0];
        tri2vtx[iquad * 6 + 1] = quad2vtx[iquad * 4 + 1];
        tri2vtx[iquad * 6 + 2] = quad2vtx[iquad * 4 + 2];
        //
        tri2vtx[iquad * 6 + 3] = quad2vtx[iquad * 4 + 0];
        tri2vtx[iquad * 6 + 4] = quad2vtx[iquad * 4 + 2];
        tri2vtx[iquad * 6 + 5] = quad2vtx[iquad * 4 + 3];
    }
    tri2vtx
}

