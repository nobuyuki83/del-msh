//! methods for 2D triangle mesh

#[allow(clippy::identity_op)]
pub fn tri2area(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32>
{
    let mut tri2area = Vec::<f32>::with_capacity(tri2vtx.len() / 3);
    for node2vtx in tri2vtx.chunks(3) {
        let (i0,i1,i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let area = del_geo::tri2::area_(
            &vtx2xyz[i0 * 2 + 0..i0 * 2 + 2],
            &vtx2xyz[i1 * 2 + 0..i1 * 2 + 2],
            &vtx2xyz[i2 * 2 + 0..i2 * 2 + 2]);
        tri2area.push(area);
    }
    tri2area
}