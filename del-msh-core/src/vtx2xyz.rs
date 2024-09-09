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

pub fn to_navec3<T>(vtx2xyz: &[T], i_vtx: usize) -> nalgebra::Vector3<T>
where
    T: Copy + nalgebra::RealField,
{
    nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz[i_vtx * 3..(i_vtx + 1) * 3])
}

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
    for i_vtx in 1..vtx2xyz.len() / 3 {
        let xyz = arrayref::array_ref!(vtx2xyz, i_vtx * 3, 3);
        del_geo_core::aabb3::update(&mut aabb, xyz, eps);
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
        del_geo_core::aabb3::update(&mut aabb, xyz, eps);
    }
    assert!(aabb[0] <= aabb[3]);
    assert!(aabb[1] <= aabb[4]);
    assert!(aabb[2] <= aabb[5]);
    aabb
}

pub trait HasXyz<Real> {
    fn xyz(&self) -> [Real;3];
}

pub fn aabb3_from_points<Real,Point: HasXyz<Real>>(points: &[Point]) -> [Real;6]
where Real: num_traits::Float
{
    let mut aabb = [Real::zero(); 6];
    {
        let xyz = points[0].xyz();
        del_geo_core::aabb3::set_as_cube(&mut aabb, &xyz, Real::zero());
    }
    for i_vtx in 1..points.len() {
        let xyz = points[i_vtx].xyz();
        del_geo_core::aabb3::update(&mut aabb, &xyz, Real::zero());
    }
    assert!(aabb[0] <= aabb[3]);
    assert!(aabb[1] <= aabb[4]);
    assert!(aabb[2] <= aabb[5]);
    aabb
}

/// Oriented Bounding Box (OBB)
pub fn obb3<Real>(vtx2xyz: &[Real]) -> [Real; 12]
where
    Real: nalgebra::RealField + Copy,
    usize: AsPrimitive<Real>,
{
    let (cov, cog) = crate::vtx2xdim::cov_cog::<Real, 3>(vtx2xyz);
    let svd = cov.svd(true, true);
    let r: nalgebra::Matrix3<Real> = svd.v_t.unwrap(); // row is the axis vectors
    let mut x_size = Real::zero();
    let mut y_size = Real::zero();
    let mut z_size = Real::zero();
    for xyz in vtx2xyz.chunks(3) {
        let p = nalgebra::Vector3::<Real>::from_row_slice(xyz);
        let l = r * (p - cog);
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
    let mut out = [Real::zero(); 12];
    out[0..3].copy_from_slice(cog.as_slice());
    out[3..6].copy_from_slice((r.row(0) * x_size).as_slice());
    out[6..9].copy_from_slice((r.row(1) * y_size).as_slice());
    out[9..12].copy_from_slice((r.row(2) * z_size).as_slice());
    out
}

#[test]
fn test_obb3() {
    let (x_size, y_size, z_size) = (0.3, 0.1, 0.5);
    let aabb3 = [-x_size, -y_size, -z_size, x_size, y_size, z_size];
    use rand::SeedableRng;
    let mut reng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let vtx2xyz0: Vec<f32> = (0..10000)
        .into_iter()
        .flat_map(|_v| del_geo_core::aabb3::sample(&aabb3, &mut reng))
        .collect();
    let obb0 = obb3(&vtx2xyz0);
    assert!(obb0[0].abs() < 0.01);
    assert!(obb0[1].abs() < 0.01);
    assert!(obb0[2].abs() < 0.01);
    //
    // let transl_vec = nalgebra::Vector3::<f32>::new(0.3, -1.0, 0.5);
    let transl_vec = nalgebra::Vector3::<f32>::new(0.3, -1.0, 0.5);
    let rot_vec = nalgebra::Vector3::<f32>::new(0.5, 0.0, 0.0);
    //
    let rot = nalgebra::Matrix4::<f32>::new_rotation(rot_vec.clone());
    let transl = nalgebra::Matrix4::<f32>::new_translation(&transl_vec);
    let mat = transl * rot;
    let vtx2xyz1 = transform(&vtx2xyz0, mat.as_slice().try_into().unwrap());
    let obb1 = obb3(&vtx2xyz1);
    assert!((obb1[0] - transl_vec[0]).abs() < 0.01);
    assert!((obb1[1] - transl_vec[1]).abs() < 0.01);
    assert!((obb1[2] - transl_vec[2]).abs() < 0.01);
    let ea = nalgebra::Vector3::<f32>::from_row_slice(&obb1[3..6]);
    let eb = nalgebra::Vector3::<f32>::from_row_slice(&obb1[6..9]);
    let ec = nalgebra::Vector3::<f32>::from_row_slice(&obb1[9..12]);
    assert!((ea.norm() - 0.5).abs() < 0.05);
    assert!((eb.norm() - 0.3).abs() < 0.05);
    assert!((ec.norm() - 0.1).abs() < 0.05);
    let dira0 = ea.normalize();
    let dira1 = nalgebra::Vector3::<f32>::from_homogeneous(rot.column(2).into_owned()).unwrap();
    assert!(dira0.cross(&dira1).norm() < 0.01);
    let dirb0 = eb.normalize();
    let dirb1 = nalgebra::Vector3::<f32>::from_homogeneous(rot.column(0).into_owned()).unwrap();
    assert!(dirb0.cross(&dirb1).norm() < 0.02);
    let dirc0 = ec.normalize();
    let dirc1 = nalgebra::Vector3::<f32>::from_homogeneous(rot.column(1).into_owned()).unwrap();
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

pub fn transform<Real>(vtx2xyz: &[Real], m: &[Real; 16]) -> Vec<Real>
where
    Real: num_traits::Float,
{
    vtx2xyz
        .chunks(3)
        .flat_map(|v| {
            del_geo_core::mat4_col_major::transform_homogeneous(m, arrayref::array_ref![v,0,3]).unwrap()
        })
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

// ------------------------------------------------

pub fn to_xyz<Real>(vtx2xyz: &[Real], i_vtx: usize) -> del_geo_core::vec3::XYZ<Real> {
    del_geo_core::vec3::XYZ {
        p: arrayref::array_ref![vtx2xyz, i_vtx * 3, 3],
    }
}
