//! methods to save files with OFF format

pub fn save_tri_mesh<P: AsRef<std::path::Path>>(
    filepath: P,
    tri2vtx: &[usize],
    vtx2xyz: &[f32])
{
    let mut file = std::fs::File::create(filepath).expect("file not found.");
    use std::io::Write;
    let num_tri = tri2vtx.len() / 3;
    let num_vtx  = vtx2xyz.len() / 3;
    writeln!(file, "OFF {} {} 0",
             num_vtx, num_tri).expect("fail");
    for i_vtx in 0..num_vtx {
        writeln!(file, "{} {} {}",
                 vtx2xyz[i_vtx * 3 + 0],
                 vtx2xyz[i_vtx * 3 + 1],
                 vtx2xyz[i_vtx * 3 + 2]).expect("fail");
    }
    for i_tri in 0..num_tri {
        writeln!(file, "3 {} {} {}",
                 tri2vtx[i_tri * 3 + 0],
                 tri2vtx[i_tri * 3 + 1],
                 tri2vtx[i_tri * 3 + 2]).expect("fail");
    }
}