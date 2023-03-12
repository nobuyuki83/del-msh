
pub fn unidex_vertex_attribute_for_triangle_mesh(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32> {
    let num_tri = tri2vtx.len() / 3;
    let mut tri2xyz = vec!(0_f32; num_tri * 9);
    for it in 0..tri2vtx.len() / 3 {
        for ino in 0..3 {
            let ip = tri2vtx[it * 3 + ino];
            tri2xyz[it * 9 + ino * 3 + 0] = vtx2xyz[ip*3+0];
            tri2xyz[it * 9 + ino * 3 + 1] = vtx2xyz[ip*3+1];
            tri2xyz[it * 9 + ino * 3 + 2] = vtx2xyz[ip*3+2];
        }
    }
    tri2xyz
}