use num_traits::AsPrimitive;

pub fn render_normalmap_from_pix2tri<INDEX>(
    (img_width, img_height): (usize, usize),
    cam_modelviewd: &[f32; 16],
    tri2vtx: &[INDEX],
    vtx2xyz: &[f32],
    pix2tri: &[INDEX],
) -> Vec<f32>
where
    INDEX: num_traits::PrimInt + AsPrimitive<usize> + Sync + Send,
{
    let mut img = vec![0f32; img_height * img_width * 3];
    for ih in 0..img_height {
        for iw in 0..img_width {
            let i_tri = pix2tri[ih * img_width + iw];
            if i_tri == INDEX::max_value() {
                continue;
            }
            let i_tri: usize = i_tri.as_();
            let tri = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri);
            let nrm = tri.normal();
            let nrm = del_geo_core::mat4_col_major::transform_direction(cam_modelviewd, &nrm);
            let unrm = del_geo_core::vec3::normalize(&nrm);
            img[(ih * img_width + iw) * 3] = unrm[0] * 0.5 + 0.5;
            img[(ih * img_width + iw) * 3 + 1] = unrm[1] * 0.5 + 0.5;
            img[(ih * img_width + iw) * 3 + 2] = unrm[2] * 0.5 + 0.5;
        }
    }
    img
}
