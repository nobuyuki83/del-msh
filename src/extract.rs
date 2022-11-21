
pub fn extract(
    tri2vtx: &[usize],
    num_vtx: usize,
    tri2tri_new: &[usize],
    num_tri_new: usize) -> (Vec<usize>, usize, Vec<usize>)
{
    assert_eq!(tri2vtx.len() / 3, tri2tri_new.len());
    let num_tri = tri2vtx.len() / 3;
    let mut vtx2vtx_new = vec!(usize::MAX; num_vtx);
    let mut tri2vtx_new = vec!(usize::MAX; num_tri_new * 3);
    let mut num_vtx_new = 0;
    for i_tri in 0..num_tri {
        if tri2tri_new[i_tri] == usize::MAX { continue; }
        let i_tri_new = tri2tri_new[i_tri];
        for i_node in 0..3 {
            let i_vtx_xyz = tri2vtx[i_tri * 3 + i_node];
            if vtx2vtx_new[i_vtx_xyz] == usize::MAX {
                let i_vtx_new = num_vtx_new;
                num_vtx_new += 1;
                vtx2vtx_new[i_vtx_xyz] = i_vtx_new;
                tri2vtx_new[i_tri_new * 3 + i_node] = i_vtx_new;
            } else {
                let i_vtx_new = vtx2vtx_new[i_vtx_xyz];
                tri2vtx_new[i_tri_new * 3 + i_node] = i_vtx_new;
            }
        }
    }
    (tri2vtx_new, num_vtx_new, vtx2vtx_new)
}

pub fn map_values_old2new(
    old2value: &[f32],
    od2new: &[usize],
    num_new: usize,
    num_dim: usize) -> Vec<f32> {
    let mut new2value = vec!(0_f32; num_new * num_dim);
    for i_old in 0..old2value.len() / num_dim {
        let i_new = od2new[i_old];
        if i_new == usize::MAX { continue; }
        for i_dim in 0..num_dim {
            new2value[i_new * num_dim + i_dim] = old2value[i_old * num_dim + i_dim];
        }
    }
    new2value
}