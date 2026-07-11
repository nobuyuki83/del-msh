use num_traits::AsPrimitive;
#[allow(clippy::too_many_arguments)]
pub fn render_texture_from_pix2tri<Index>(
    img_shape: (usize, usize),
    transform_ndc2world: &[f32; 16],
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    vtx2uv: &[f32],
    pix2tri: &[Index],
    tex_shape: (usize, usize),
    tex_data: &[f32],
    interpolation: &crate::grid2::Interpolation,
) -> Vec<f32>
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    let (width, height) = img_shape;
    let mut img = vec![0f32; height * width * 3];
    for ih in 0..height {
        for iw in 0..width {
            let (ray_org, ray_dir) =
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinates(
                    (iw as f32 + 0.5, ih as f32 + 0.5),
                    &(img_shape.0 as f32, img_shape.1 as f32),
                    transform_ndc2world,
                );
            let i_tri = pix2tri[ih * width + iw];
            if i_tri == Index::max_value() {
                continue;
            }
            let i_tri: usize = i_tri.as_();
            let tri = crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri);
            let Some((a, _bc)) = tri.intersection_against_ray(&ray_org, &ray_dir) else {
                continue;
            };
            let q = del_geo_core::vec3::axpy(a, &ray_dir, &ray_org);
            let bc = del_geo_core::tri3::to_barycentric_coords(tri.p0, tri.p1, tri.p2, &q);
            let uv0 = arrayref::array_ref!(vtx2uv, tri2vtx[i_tri * 3] * 2, 2);
            let uv1 = arrayref::array_ref!(vtx2uv, tri2vtx[i_tri * 3 + 1] * 2, 2);
            let uv2 = arrayref::array_ref!(vtx2uv, tri2vtx[i_tri * 3 + 2] * 2, 2);
            let uv = [
                uv0[0] * bc[0] + uv1[0] * bc[1] + uv2[0] * bc[2],
                uv0[1] * bc[0] + uv1[1] * bc[1] + uv2[1] * bc[2],
            ];
            let pix = [
                uv[0] * tex_shape.0 as f32,
                (1. - uv[1]) * tex_shape.1 as f32,
            ];
            let res = match interpolation {
                crate::grid2::Interpolation::Nearest => {
                    crate::grid2::nearest_integer_center::<3>(&pix, &tex_shape, tex_data)
                }
                crate::grid2::Interpolation::Bilinear => {
                    crate::grid2::bilinear_integer_center::<3>(&pix, &tex_shape, tex_data)
                }
            };
            img[(ih * width + iw) * 3] = res[0];
            img[(ih * width + iw) * 3 + 1] = res[1];
            img[(ih * width + iw) * 3 + 2] = res[2];
        }
    }
    img
}
