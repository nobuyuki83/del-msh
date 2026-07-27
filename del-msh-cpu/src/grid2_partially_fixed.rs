

pub fn smooth_gauss_seidel(
    (w, h): (usize, usize),
    cell2isfix: &[u8],
    num_vdim: usize,
    cell2val: &mut [f32],
) {
    assert_eq!(cell2isfix.len(), h * w);
    assert_eq!(cell2isfix.len() * num_vdim, cell2val.len());
    let mut v_sum = vec!(0f32; num_vdim);
    for i_cell in 0..h * w {
        let iw = i_cell % w;
        let ih = i_cell / w;
        if cell2isfix[i_cell] == 1 {
            continue;
        }
        v_sum.fill(0.);
        let mut n_sum = 0;
        //
        if ih != 0 { // north
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[((ih - 1) * w + iw) * num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        if ih != h - 1 { // south
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[((ih + 1) * w + iw)* num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        if iw != 0 { // west
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[(ih * w + iw - 1)*num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        if iw != w - 1 { // east
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[(ih * w + iw + 1)*num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        // ----------------------------
        if n_sum != 0 {
            let t = 1.0 / n_sum as f32;
            for i_vdim in 0..num_vdim {
                cell2val[i_cell * num_vdim + i_vdim] = v_sum[i_vdim] * t;
            }
        }
    }
}

pub fn smooth_gauss_seidel_with_radius(
    (w, h): (usize, usize),
    cell2isfixed: &[u8],
    cell2dist: &[f32],
    ratio: f32,
    num_vdim: usize,
    cell2val: &mut [f32],
) {
    assert_eq!(cell2isfixed.len(), h * w);
    assert_eq!(cell2dist.len(), h * w);
    assert_eq!(cell2isfixed.len() * num_vdim, cell2val.len());
    let mut v_sum = vec!(0f32; num_vdim);
    for i_cell in 0..h * w {
        let iw1 = (i_cell % w) as i64;
        let ih1 = (i_cell / w) as i64;
        let dist = cell2dist[i_cell];
        let r = (dist * ratio).floor() as i64;
        let (iw0, iw2) = (iw1 - r, iw1 + r);
        let (ih0, ih2) = (ih1 - r, ih1 + r);
        if cell2isfixed[i_cell] == 1 {
            continue;
        }
        v_sum.fill(0.);
        let mut n_sum = 0;
        //
        if ih0 >= 0 { // north
            let (ih0, iw1) = (ih0 as usize, iw1 as usize);
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[(ih0 * w + iw1) * num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        if ih2 < h as i64 { // south
            let (ih2, iw1) = (ih2 as usize, iw1 as usize);
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[(ih2 * w + iw1)* num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        if iw0 >= 0 { // west
            let (ih1, iw0) = (ih1 as usize, iw0 as usize);
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[(ih1 * w + iw0)*num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        if iw2 < w as i64 { // east
            let (ih1, iw2) = (ih1 as usize, iw2 as usize);
            for i_vdim in 0..num_vdim {
                v_sum[i_vdim] += cell2val[(ih1 * w + iw2)*num_vdim + i_vdim];
            }
            n_sum += 1;
        }
        // ----------------------------
        if n_sum != 0 {
            let t = 1.0 / n_sum as f32;
            for i_vdim in 0..num_vdim {
                cell2val[i_cell * num_vdim + i_vdim] = v_sum[i_vdim] * t;
            }
        }
    }
}


// 2-pass chamfer transform for 8 adjacent cells
pub fn nearest_to_fixed_cell(
    (w, h): (usize, usize),
    cell2isfixed: &[u8],
    cell2nearest: &mut [u32],
    cell2distance: &mut [f32],
    is_initial: bool,
) {
    assert_eq!(cell2isfixed.len(), w * h);
    assert_eq!(cell2isfixed.len(), cell2nearest.len());
    assert_eq!(cell2isfixed.len(), cell2distance.len());
    //

    if w == 0 || h == 0 {
        return;
    }

    const INVALID: u32 = u32::MAX;

    assert!(
        w * h < u32::MAX as usize,
        "The number of cells must fit in u32"
    );

    if is_initial {
        // During the sweeps, cell2distance stores squared distance.
        for i_cell in 0..w * h {
            if cell2isfixed[i_cell] != 0 {
                cell2nearest[i_cell] = i_cell as u32;
                cell2distance[i_cell] = 0.0;
            } else {
                cell2nearest[i_cell] = INVALID;
                cell2distance[i_cell] = f32::INFINITY;
            }
        }
    }

    #[inline]
    fn try_update(
        i_cell: usize,
        i_neighbor: usize,
        w: usize,
        cell2nearest: &mut [u32],
        cell2distance2: &mut [f32],
    ) {
        const INVALID: u32 = u32::MAX;

        let i_seed = cell2nearest[i_neighbor];
        if i_seed == INVALID {
            return;
        }

        let i_seed = i_seed as usize;

        let ix = i_cell % w;
        let iy = i_cell / w;

        let ix_seed = i_seed % w;
        let iy_seed = i_seed / w;

        let dx = ix.abs_diff(ix_seed) as f32;
        let dy = iy.abs_diff(iy_seed) as f32;
        let distance2 = dx * dx + dy * dy;

        if distance2 < cell2distance2[i_cell] {
            cell2distance2[i_cell] = distance2;
            cell2nearest[i_cell] = i_seed as u32;
        }
    }

    // Forward sweep:
    //
    // NW  N  NE
    //  W  X
    for iy in 0..h {
        for ix in 0..w {
            let i_cell = iy * w + ix;

            if cell2isfixed[i_cell] != 0 {
                continue;
            }

            // West
            if ix > 0 {
                try_update(
                    i_cell,
                    i_cell - 1,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }

            // North-west
            if ix > 0 && iy > 0 {
                try_update(
                    i_cell,
                    i_cell - w - 1,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }

            // North
            if iy > 0 {
                try_update(
                    i_cell,
                    i_cell - w,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }

            // North-east
            if ix + 1 < w && iy > 0 {
                try_update(
                    i_cell,
                    i_cell - w + 1,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }
        }
    }

    // Backward sweep:
    //
    //       X  E
    // SW  S  SE
    for iy in (0..h).rev() {
        for ix in (0..w).rev() {
            let i_cell = iy * w + ix;

            if cell2isfixed[i_cell] != 0 {
                continue;
            }

            // East
            if ix + 1 < w {
                try_update(
                    i_cell,
                    i_cell + 1,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }

            // South-east
            if ix + 1 < w && iy + 1 < h {
                try_update(
                    i_cell,
                    i_cell + w + 1,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }

            // South
            if iy + 1 < h {
                try_update(
                    i_cell,
                    i_cell + w,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }

            // South-west
            if ix > 0 && iy + 1 < h {
                try_update(
                    i_cell,
                    i_cell + w - 1,
                    w,
                    cell2nearest,
                    cell2distance,
                );
            }
        }
    }

    // Convert squared distances to Euclidean distances.
    for distance2 in cell2distance {
        *distance2 = distance2.sqrt();
    }
}

#[test]
fn test_nearest_to_fixed_cell(){
    let img_shape = (512,511);
    let num_pix = img_shape.0 * img_shape.1;
    let mut pix2rgb = vec!(0f32; num_pix * 3);
    let mut pix2isfix = vec!(0u8; num_pix);
    let n_fix = 100;
    use rand::RngExt;
    use rand::SeedableRng;
    let rng = &mut rand::rngs::StdRng::seed_from_u64(0);
    for _i_fix in 0..n_fix {
        let i_pix = rng.random_range(0..(img_shape.0*img_shape.1) as usize);
        pix2rgb[i_pix*3] = rng.random();
        pix2rgb[i_pix*3+1] = rng.random();
        pix2rgb[i_pix*3+2] = rng.random();
        pix2isfix[i_pix] = 1;
    }
    {
        let mut pix2rgb1 = pix2rgb.clone();
        for i_gs in 0..8 {
            del_canvas::write_png_from_float_image(
                format!("../target/grid2_partially_fixed_gs{i_gs}.png"),
                img_shape,
                3,
                &pix2rgb1
            ).unwrap();
            for _iter in 0..100 {
                smooth_gauss_seidel(
                    img_shape,
                    &pix2isfix,
                    3,
                    &mut pix2rgb1);
            }
        }
    }
    {
        let mut pix2nearest = vec!(0u32; num_pix);
        let mut pix2distance = vec!(0f32; num_pix);
        nearest_to_fixed_cell(
            img_shape,
            &pix2isfix,
            &mut pix2nearest,
            &mut pix2distance,
            true
        );
        let mut pix2rgb1 = vec!(0f32; num_pix * 3);
        for i_pix in 0..num_pix {
            let i_pix_near = pix2nearest[i_pix] as usize;
            for i_vdim in 0..3 {
                pix2rgb1[i_pix*3+i_vdim] = pix2rgb[i_pix_near*3+i_vdim];
            }
        }
        let n_iter = 8;
        for iter in 0..n_iter {
            del_canvas::write_png_from_float_image(
                format!("../target/grid2_partially_fixed_jump{iter}.png"),
                img_shape,
                3,
                &pix2rgb1
            ).unwrap();
            let ratio = 1.0 - (iter + 1) as f32 / n_iter as f32;
            smooth_gauss_seidel_with_radius(
                img_shape,
                &pix2isfix,
                &pix2distance,
                ratio,
                3,
                &mut pix2rgb1
            );
        }

    }
}

