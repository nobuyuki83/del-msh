pub fn elem2volume(
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    vtx2xyz: &[f32],
    i_gauss_degree: usize,
    elem2volume: &mut [f32],
) {
    let num_elem = elem2idx_offset.len() - 1;
    assert_eq!(elem2volume.len(), num_elem);
    for i_elem in 0..elem2idx_offset.len() - 1 {
        let node2vtx =
            &idx2vtx[elem2idx_offset[i_elem] as usize..elem2idx_offset[i_elem + 1] as usize];
        match node2vtx.len() {
            4 => {
                let p0 = arrayref::array_ref![vtx2xyz, node2vtx[0] as usize * 3, 3];
                let p1 = arrayref::array_ref![vtx2xyz, node2vtx[1] as usize * 3, 3];
                let p2 = arrayref::array_ref![vtx2xyz, node2vtx[2] as usize * 3, 3];
                let p3 = arrayref::array_ref![vtx2xyz, node2vtx[3] as usize * 3, 3];
                elem2volume[i_elem] = del_geo_core::tet::volume(p0, p1, p2, p3);
            }
            5 => {
                let p0 = arrayref::array_ref![vtx2xyz, node2vtx[0] as usize * 3, 3];
                let p1 = arrayref::array_ref![vtx2xyz, node2vtx[1] as usize * 3, 3];
                let p2 = arrayref::array_ref![vtx2xyz, node2vtx[2] as usize * 3, 3];
                let p3 = arrayref::array_ref![vtx2xyz, node2vtx[3] as usize * 3, 3];
                let p4 = arrayref::array_ref![vtx2xyz, node2vtx[4] as usize * 3, 3];
                elem2volume[i_elem] =
                    del_geo_core::pyramid::volume(p0, p1, p2, p3, p4, i_gauss_degree);
            }
            6 => {
                let p0 = arrayref::array_ref![vtx2xyz, node2vtx[0] as usize * 3, 3];
                let p1 = arrayref::array_ref![vtx2xyz, node2vtx[1] as usize * 3, 3];
                let p2 = arrayref::array_ref![vtx2xyz, node2vtx[2] as usize * 3, 3];
                let p3 = arrayref::array_ref![vtx2xyz, node2vtx[3] as usize * 3, 3];
                let p4 = arrayref::array_ref![vtx2xyz, node2vtx[4] as usize * 3, 3];
                let p5 = arrayref::array_ref![vtx2xyz, node2vtx[5] as usize * 3, 3];
                elem2volume[i_elem] =
                    del_geo_core::prism::volume(p0, p1, p2, p3, p4, p5, i_gauss_degree);
            }
            _ => {
                unreachable!()
            }
        }
    }
}
