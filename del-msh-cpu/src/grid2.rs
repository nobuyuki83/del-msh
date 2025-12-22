pub enum Interpolation {
    Nearest,
    Bilinear,
}

/// coordinate (0., 0.) is the center ot the texel
pub fn bilinear_integer_center<const NDIM: usize>(
    pix: &[f32; 2],
    tex_shape: &(usize, usize),
    tex_data: &[f32],
) -> [f32; NDIM] {
    let rx = pix[0] - pix[0].floor();
    let ix0 = pix[0].floor() as i64;
    if ix0 < 0 || ix0 >= tex_shape.0 as i64 {
        return [0f32; NDIM];
    }
    let ry = pix[1] - pix[1].floor();
    let iy0 = pix[1].floor() as i64;
    if iy0 < 0 || iy0 >= tex_shape.1 as i64 {
        return [0f32; NDIM];
    }
    //
    let ix0 = ix0 as usize;
    let iy0 = iy0 as usize;
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;
    let i00_tex = iy0 * tex_shape.0 + ix0;
    let i10_tex = iy0 * tex_shape.0 + ix1;
    let i01_tex = iy1 * tex_shape.0 + ix0;
    let i11_tex = iy1 * tex_shape.0 + ix1;
    std::array::from_fn(|i_dim| {
        let v00 = tex_data[i00_tex * NDIM + i_dim];
        let v01 = tex_data[i01_tex * NDIM + i_dim];
        let v10 = tex_data[i10_tex * NDIM + i_dim];
        let v11 = tex_data[i11_tex * NDIM + i_dim];
        (1. - rx) * (1. - ry) * v00 + rx * (1. - ry) * v10 + (1. - rx) * ry * v01 + rx * ry * v11
    })
}

/// coordinate (0., 0.) is the center ot the texel
pub fn nearest_integer_center<const NDIM: usize>(
    pix: &[f32; 2],
    tex_shape: &(usize, usize),
    tex_data: &[f32],
) -> [f32; NDIM] {
    let ix0 = pix[0].round() as i64;
    if ix0 < 0 || ix0 >= tex_shape.0 as i64 {
        return [0f32; NDIM];
    }
    let iy0 = pix[1].round() as i64;
    if iy0 < 0 || iy0 >= tex_shape.1 as i64 {
        return [0f32; NDIM];
    }
    let i_tex = (iy0 as usize) * tex_shape.0 + (ix0 as usize);
    // assert!(i_tex >=0 && i_tex < tex_shape.1);
    std::array::from_fn(|i_dim| tex_data[i_tex * NDIM + i_dim])
}

// left up corner is the origin
pub fn to_quadmesh3_hightmap(
    grid_shape: (usize, usize),
    height: &[f32],
    elen: f32,
) -> (Vec<usize>, Vec<f32>) {
    let nw = grid_shape.0;
    let nh = grid_shape.1;
    let _num_vtx = nw * nh;
    let mw = nw - 1; // quads in width
    let mh = nh - 1; // quads in height
    let mut quad2vtx = Vec::<usize>::with_capacity(mw * mh * 4);
    for ih in 0..mh {
        for iw in 0..mw {
            let i01_vtx = ih * nw + iw;
            let i11_vtx = ih * nw + iw + 1;
            let i00_vtx = (ih + 1) * nw + iw;
            let i10_vtx = (ih + 1) * nw + iw + 1;
            quad2vtx.extend_from_slice(&[i00_vtx, i10_vtx, i11_vtx, i01_vtx]);
        }
    }
    let mut vtx2xyz = Vec::<f32>::with_capacity(nw * nh * 3);
    for ih in 0..nh {
        for iw in 0..nw {
            let x = iw as f32 * elen;
            let y = ih as f32 * -elen;
            let z = height[ih * nw + iw];
            vtx2xyz.extend_from_slice(&[x, y, z]);
        }
    }
    (quad2vtx, vtx2xyz)
}
