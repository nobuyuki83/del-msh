//! methods for 2D triangle mesh

pub fn areas(
    tri2vtx: &[usize],
    vtx2xyz: &[f32], ) -> Vec<f32>
{
    let mut tri2area = vec!();
    tri2area.reserve(tri2vtx.len() / 3);
    for i_vtxs in tri2vtx.chunks(3) {
        let (i0,i1,i2) = (i_vtxs[0], i_vtxs[1], i_vtxs[2]);
        let area = del_geo::tri2::area_(
            &vtx2xyz[i0 * 2 + 0..i0 * 2 + 2],
            &vtx2xyz[i1 * 2 + 0..i1 * 2 + 2],
            &vtx2xyz[i2 * 2 + 0..i2 * 2 + 2]);
        tri2area.push(area);
    }
    tri2area
}