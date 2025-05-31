//! functions related to c-style contiguous array of 3D coordinates

use num_traits::AsPrimitive;

pub fn to_array3<T>(vtx2xyz: &[T], i_vtx: usize) -> [T; 3]
where
    T: Copy,
{
    [
        vtx2xyz[i_vtx * 3],
        vtx2xyz[i_vtx * 3 + 1],
        vtx2xyz[i_vtx * 3 + 2],
    ]
}

/// 3D Axis-aligned bonding box for 3D points
/// # Arguments
/// * eps: T - margin
pub fn aabb3<T>(vtx2xyz: &[T], eps: T) -> [T; 6]
where
    T: num_traits::Float,
{
    assert!(!vtx2xyz.is_empty());
    let mut aabb = [T::zero(); 6];
    {
        let xyz = arrayref::array_ref!(vtx2xyz, 0, 3);
        del_geo_core::aabb3::set_as_cube(&mut aabb, xyz, eps);
    }
    for xyz in vtx2xyz.chunks(3) {
        let xyz = arrayref::array_ref!(xyz, 0, 3);
        del_geo_core::aabb3::add_point(&mut aabb, xyz, eps);
    }
    assert!(aabb[0] <= aabb[3]);
    assert!(aabb[1] <= aabb[4]);
    assert!(aabb[2] <= aabb[5]);
    aabb
}

pub fn aabb3_indexed<Index, Real>(idx2vtx: &[Index], vtx2xyz: &[Real], eps: Real) -> [Real; 6]
where
    Real: num_traits::Float,
    Index: AsPrimitive<usize>,
{
    assert!(!idx2vtx.is_empty());
    let mut aabb = [Real::zero(); 6];
    {
        let i_vtx: usize = idx2vtx[0].as_();
        let xyz = arrayref::array_ref!(vtx2xyz, i_vtx * 3, 3);
        del_geo_core::aabb3::set_as_cube(&mut aabb, xyz, eps);
    }
    for &i_vtx in idx2vtx.iter().skip(1) {
        let i_vtx: usize = i_vtx.as_();
        let xyz = arrayref::array_ref!(vtx2xyz, i_vtx * 3, 3);
        del_geo_core::aabb3::add_point(&mut aabb, xyz, eps);
    }
    assert!(aabb[0] <= aabb[3]);
    assert!(aabb[1] <= aabb[4]);
    assert!(aabb[2] <= aabb[5]);
    aabb
}

/// Oriented Bounding Box (OBB)
pub fn obb3<Real>(vtx2xyz: &[Real]) -> [Real; 12]
where
    Real: num_traits::Float + Copy + 'static + std::iter::Sum + num_traits::FloatConst,
    usize: AsPrimitive<Real>,
{
    use del_geo_core::vec3::Vec3;
    let (cov, cog) = crate::vtx2xn::cov_cog::<Real, 3>(vtx2xyz);
    use slice_of_array::SliceFlatExt;
    let cov = cov.flat();
    let cov = arrayref::array_ref![cov, 0, 9];
    let (_u, _s, v) = del_geo_core::mat3_row_major::svd(
        cov,
        del_geo_core::mat3_sym::EigenDecompositionModes::Analytic,
    )
    .unwrap();
    // let r: nalgebra::Matrix3<Real> = svd.v_t.unwrap(); // row is the axis vectors
    let mut x_size = Real::zero();
    let mut y_size = Real::zero();
    let mut z_size = Real::zero();
    for xyz in vtx2xyz.chunks(3) {
        let xyz = arrayref::array_ref![xyz, 0, 3];
        let l = del_geo_core::mat3_col_major::mult_vec(&v, &xyz.sub(&cog)); // v^t * (v-cog)
        x_size = if l[0].abs() > x_size {
            l[0].abs()
        } else {
            x_size
        };
        y_size = if l[1].abs() > y_size {
            l[1].abs()
        } else {
            y_size
        };
        z_size = if l[2].abs() > z_size {
            l[2].abs()
        } else {
            z_size
        };
    }
    let (v0, v1, v2) = del_geo_core::mat3_row_major::to_columns(&v);
    let mut out = [Real::zero(); 12];
    out[0..3].copy_from_slice(cog.as_slice());
    out[3..6].copy_from_slice(&v0.scale(x_size));
    out[6..9].copy_from_slice(&v1.scale(y_size));
    out[9..12].copy_from_slice(&v2.scale(z_size));
    out
}

#[test]
fn test_obb3() {
    use del_geo_core::vec3::Vec3;
    let (x_size, y_size, z_size) = (0.3, 0.1, 0.5);
    let aabb3 = [-x_size, -y_size, -z_size, x_size, y_size, z_size];
    use rand::SeedableRng;
    let mut reng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let vtx2xyz0: Vec<f32> = (0..10000)
        .flat_map(|_v| del_geo_core::aabb3::sample(&aabb3, &mut reng))
        .collect();
    let obb0 = obb3(&vtx2xyz0);
    assert!(obb0[0].abs() < 0.01);
    assert!(obb0[1].abs() < 0.01);
    assert!(obb0[2].abs() < 0.01);
    //
    let transl_vec = [0.3, -1.0, 0.5];
    let transl = del_geo_core::mat4_col_major::from_translate(&transl_vec);
    let rot = del_geo_core::mat4_col_major::from_bryant_angles(0.5, 0.0, 0.0);
    let mat = del_geo_core::mat4_col_major::mult_mat_col_major(&transl, &rot);
    let vtx2xyz1 = transform_homogeneous(&vtx2xyz0, &mat);
    let obb1 = obb3(&vtx2xyz1);
    assert!((obb1[0] - transl_vec[0]).abs() < 0.01);
    assert!((obb1[1] - transl_vec[1]).abs() < 0.01);
    assert!((obb1[2] - transl_vec[2]).abs() < 0.01);
    let (ea, eb, ec) = {
        let ea = arrayref::array_ref![&obb1, 3, 3];
        let eb = arrayref::array_ref![&obb1, 6, 3];
        let ec = arrayref::array_ref![&obb1, 9, 3];
        let mut a = [(ea, ea.norm()), (eb, eb.norm()), (ec, ec.norm())];
        a.sort_by(|&(_a0, a1), &(_b0, b1)| a1.partial_cmp(&b1).unwrap());
        (a[2].0, a[1].0, a[0].0)
    };
    assert!((ea.norm() - 0.5).abs() < 0.05);
    assert!((eb.norm() - 0.3).abs() < 0.05);
    assert!((ec.norm() - 0.1).abs() < 0.05);
    let rot3 = del_geo_core::mat4_col_major::to_mat3_col_major_xyz(&rot);
    let (dirb1, dirc1, dira1) = del_geo_core::mat3_col_major::to_columns(&rot3);
    let dira0 = ea.normalize();
    assert!(dira0.cross(&dira1).norm() < 0.01);
    let dirb0 = eb.normalize();
    assert!(dirb0.cross(&dirb1).norm() < 0.02);
    let dirc0 = ec.normalize();
    assert!(dirc0.cross(&dirc1).norm() < 0.01);
    // check all the points are included
    for xyz in vtx2xyz1.chunks(3) {
        let is_include = del_geo_core::obb3::is_include_point(&obb1, xyz.try_into().unwrap(), 0.01);
        assert!(is_include);
    }
}

// above: to_****
// ---------------------

pub fn translate_then_scale<Real>(
    vtx2xyz_out: &mut [Real],
    vtx2xyz_in: &[Real],
    transl: &[Real; 3],
    scale: Real,
) where
    Real: num_traits::Float,
{
    assert_eq!(vtx2xyz_in.len(), vtx2xyz_out.len());
    vtx2xyz_out
        .chunks_mut(3)
        .zip(vtx2xyz_in.chunks(3))
        .for_each(|(o, v)| {
            o[0] = (v[0] + transl[0]) * scale;
            o[1] = (v[1] + transl[1]) * scale;
            o[2] = (v[2] + transl[2]) * scale;
        });
    /*
    let num_vtx = vtx2xyz_in.len() / 3;
    for i_vtx in 0..num_vtx {
        let p = &vtx2xyz_in[i_vtx * 3..(i_vtx + 1) * 3];
        let q = &mut vtx2xyz_out[i_vtx * 3..(i_vtx + 1) * 3];
        q[0] = (p[0] + transl[0]) * scale;
        q[1] = (p[1] + transl[1]) * scale;
        q[2] = (p[2] + transl[2]) * scale;
    }
     */
}

pub fn transform_homogeneous<Real>(vtx2xyz: &[Real], m: &[Real; 16]) -> Vec<Real>
where
    Real: num_traits::Float,
{
    vtx2xyz
        .chunks(3)
        .flat_map(|v| {
            del_geo_core::mat4_col_major::transform_homogeneous(m, arrayref::array_ref![v, 0, 3])
                .unwrap()
        })
        .collect()
}

pub fn transform_linear<Real>(vtx2xyz: &[Real], m: &[Real; 9]) -> Vec<Real>
where
    Real: num_traits::Float,
{
    vtx2xyz
        .chunks(3)
        .flat_map(|v| del_geo_core::mat3_col_major::mult_vec(m, arrayref::array_ref![v, 0, 3]))
        .collect()
}

pub fn normalize<Real>(vtx2xyz: &[Real]) -> Vec<Real>
where
    Real: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<Real>,
{
    let aabb = aabb3(vtx2xyz, Real::zero());
    let cnt = del_geo_core::aabb3::center(&aabb);
    let transl = [-cnt[0], -cnt[1], -cnt[2]];
    let max_edge_size = del_geo_core::aabb3::max_edge_size(&aabb);
    let scale = Real::one() / max_edge_size;
    let mut vtx2xyz_out = Vec::from(vtx2xyz);
    translate_then_scale(&mut vtx2xyz_out, vtx2xyz, &transl, scale);
    vtx2xyz_out
}

#[allow(clippy::identity_op)]
pub fn normalize_in_place<Real>(vtx2xyz: &mut [Real], size: Real)
where
    Real: num_traits::Float,
{
    let num_vtx = vtx2xyz.len() / 3;
    let mut mins = [Real::one(); 3];
    let mut maxs = [-Real::one(); 3];
    for ivtx in 0..num_vtx {
        let x0 = vtx2xyz[ivtx * 3 + 0];
        let y0 = vtx2xyz[ivtx * 3 + 1];
        let z0 = vtx2xyz[ivtx * 3 + 2];
        if ivtx == 0 {
            mins[0] = x0;
            maxs[0] = x0;
            mins[1] = y0;
            maxs[1] = y0;
            mins[2] = z0;
            maxs[2] = z0;
        } else {
            mins[0] = if x0 < mins[0] { x0 } else { mins[0] };
            maxs[0] = if x0 > maxs[0] { x0 } else { maxs[0] };
            mins[1] = if y0 < mins[1] { y0 } else { mins[1] };
            maxs[1] = if y0 > maxs[1] { y0 } else { maxs[1] };
            mins[2] = if z0 < mins[2] { z0 } else { mins[2] };
            maxs[2] = if z0 > maxs[2] { z0 } else { maxs[2] };
        }
    }
    let half = Real::one() / (Real::one() + Real::one());
    let cntr = [
        (mins[0] + maxs[0]) * half,
        (mins[1] + maxs[1]) * half,
        (mins[2] + maxs[2]) * half,
    ];
    let scale = {
        let mut size0 = maxs[0] - mins[0];
        if maxs[1] - mins[1] > size0 {
            size0 = maxs[1] - mins[1];
        }
        if maxs[2] - mins[2] > size0 {
            size0 = maxs[2] - mins[2];
        }
        size / size0
    };
    for ivtx in 0..num_vtx {
        let x0 = vtx2xyz[ivtx * 3 + 0];
        let y0 = vtx2xyz[ivtx * 3 + 1];
        let z0 = vtx2xyz[ivtx * 3 + 2];
        vtx2xyz[ivtx * 3 + 0] = (x0 - cntr[0]) * scale;
        vtx2xyz[ivtx * 3 + 1] = (y0 - cntr[1]) * scale;
        vtx2xyz[ivtx * 3 + 2] = (z0 - cntr[2]) * scale;
    }
}

// ------------------------------------------------

pub fn to_xyz<Real>(vtx2xyz: &[Real], i_vtx: usize) -> del_geo_core::vec3::XYZ<Real> {
    del_geo_core::vec3::XYZ {
        p: arrayref::array_ref![vtx2xyz, i_vtx * 3, 3],
    }
}

pub fn to_vec3<Real>(vtx2xyz: &[Real], i_vtx: usize) -> &[Real; 3] {
    arrayref::array_ref![vtx2xyz, i_vtx * 3, 3]
}

pub fn to_vec3_mut<Real>(vtx2xyz: &mut [Real], i_vtx: usize) -> &mut [Real; 3] {
    arrayref::array_mut_ref![vtx2xyz, i_vtx * 3, 3]
}
