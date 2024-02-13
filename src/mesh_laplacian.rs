use num_traits::AsPrimitive;

pub fn mesh_laplacian_cotangent<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    row2val: &mut [T],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>)
    where T: num_traits::Float + 'static + std::ops::AddAssign,
          f64: num_traits::AsPrimitive<T>
{
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let v0: &[T; 3] = &vtx2xyz[i0 * 3..i0 * 3 + 3].try_into().unwrap();
        let v1: &[T; 3] = &vtx2xyz[i1 * 3..i1 * 3 + 3].try_into().unwrap();
        let v2: &[T; 3] = &vtx2xyz[i2 * 3..i2 * 3 + 3].try_into().unwrap();
        let emat: [T; 9] = del_geo::tri3::emat_cotangent_laplacian(v0, v1, v2);
        crate::merge(
            node2vtx, node2vtx, &emat,
            row2idx, idx2col,
            row2val, idx2val,
            merge_buffer);
    }
}

pub fn optimal_rotation_for_arap_spoke<T>(
    i_vtx: usize,
    adj2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    adj2weight: &[T],
    weight_scale: T) -> nalgebra::Matrix3::<T>
    where T: nalgebra::RealField + Copy + std::ops::AddAssign
{
    let p0 = &vtx2xyz_ini[i_vtx * 3..i_vtx * 3 + 3].try_into().unwrap();
    let p1 = &vtx2xyz_def[i_vtx * 3..i_vtx * 3 + 3].try_into().unwrap();
    let mut a = nalgebra::Matrix3::<T>::zeros();
    for idx in 0..adj2vtx.len() {
        let j_vtx = adj2vtx[idx];
        let q0 = &vtx2xyz_ini[j_vtx * 3..j_vtx * 3 + 3].try_into().unwrap();
        let q1 = &vtx2xyz_def[j_vtx * 3..j_vtx * 3 + 3].try_into().unwrap();
        let pq0 = del_geo::vec3::sub_(q0, p0);
        let pq1 = del_geo::vec3::sub_(q1, p1);
        let w = adj2weight[idx] * weight_scale;
        a.m11 += w * pq1[0] * pq0[0];
        a.m12 += w * pq1[0] * pq0[1];
        a.m13 += w * pq1[0] * pq0[2];
        a.m21 += w * pq1[1] * pq0[0];
        a.m22 += w * pq1[1] * pq0[1];
        a.m23 += w * pq1[1] * pq0[2];
        a.m31 += w * pq1[2] * pq0[0];
        a.m32 += w * pq1[2] * pq0[1];
        a.m33 += w * pq1[2] * pq0[2];
    }
    del_geo::mat3::rotational_component(&a)
}

#[test]
fn test_optimal_rotation_for_arap() {
    let (tri2vtx, vtx2xyz_ini)
        = crate::trimesh3_primitive::capsule_yup(
        0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let (row2idx, idx2col)
        = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx);
    let (_row2val, idx2val) = {
        let mut row2val = vec!(0f64; num_vtx);
        let mut idx2val = vec!(0f64; idx2col.len());
        let mut merge_buffer = vec!(0usize; 0);
        mesh_laplacian_cotangent(
            &tri2vtx, &vtx2xyz_ini,
            &row2idx, &idx2col,
            &mut row2val, &mut idx2val, &mut merge_buffer);
        (row2val, idx2val)
    };
    let mut vtx2xyz_def = vtx2xyz_ini.clone();
    let r0 = {
        let a_mat = nalgebra::Matrix4::<f64>::new_rotation(
            nalgebra::Vector3::<f64>::new(1., 2., 3.));
        for i_vtx in 0..vtx2xyz_def.len() / 3 {
            let p0 = nalgebra::Vector3::<f64>::new(
                vtx2xyz_ini[i_vtx * 3 + 0],
                vtx2xyz_ini[i_vtx * 3 + 1],
                vtx2xyz_ini[i_vtx * 3 + 2]);
            let p1 = a_mat.transform_vector(&p0);
            vtx2xyz_def[i_vtx * 3 + 0] = p1.x;
            vtx2xyz_def[i_vtx * 3 + 1] = p1.y;
            vtx2xyz_def[i_vtx * 3 + 2] = p1.z;
        }
        let r0: nalgebra::Matrix3::<f64> = a_mat.fixed_view::<3, 3>(0, 0).into();
        r0
    };
    for i_vtx in 0..vtx2xyz_ini.len() / 3 {
        let r = optimal_rotation_for_arap_spoke(
            i_vtx,
            &idx2col[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            &vtx2xyz_ini,
            &vtx2xyz_def,
            &idx2val[row2idx[i_vtx]..row2idx[i_vtx + 1]], -1.);
        assert!((r.determinant() - 1.0).abs() < 1.0e-5);
        assert!((r - r0).norm() < 1.0e-5);
    }
}

fn energy_par_vtx_arap_spoke<T>(
    i_vtx: usize,
    adj2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    adj2weight: &[T],
    weight_scale: T,
    rot_mat: &nalgebra::Matrix3::<T>) -> T
    where T: nalgebra::RealField + Copy + std::ops::AddAssign
{
    let p0 = &vtx2xyz_ini[i_vtx * 3..i_vtx * 3 + 3].try_into().unwrap();
    let p1 = &vtx2xyz_def[i_vtx * 3..i_vtx * 3 + 3].try_into().unwrap();
    let mut w = T::zero();
    for idx in 0..adj2vtx.len() {
        let j_vtx = adj2vtx[idx];
        let q0 = &vtx2xyz_ini[j_vtx * 3..j_vtx * 3 + 3].try_into().unwrap();
        let q1 = &vtx2xyz_def[j_vtx * 3..j_vtx * 3 + 3].try_into().unwrap();
        let pq0 = del_geo::vec3::sub_(q0, p0);
        let pq1 = del_geo::vec3::sub_(q1, p1);
        let pq0 = rot_mat * nalgebra::Vector3::<T>::from(pq0);
        let pq1 = nalgebra::Vector3::<T>::from(pq1);
        let diff = (pq1 - pq0).norm_squared();
        w += adj2weight[idx] * weight_scale * diff;
    }
    w
}

pub fn energy_arap_spoke<T>(
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    weight_scale: T,
    idx2val: &[T],
    vtx2rot: &[T]) -> T
    where T: nalgebra::RealField + Copy + std::ops::AddAssign
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    let mut tot_w = T::zero();
    for i_vtx in 0..num_vtx {
        let r0 = nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i_vtx * 9..i_vtx * 9 + 9]);
        let i_w = energy_par_vtx_arap_spoke(
            i_vtx,
            &idx2col[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            vtx2xyz_ini,
            vtx2xyz_def,
            &idx2val[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            weight_scale,
            &r0);
        tot_w += i_w;
    }
    tot_w
}


#[test]
fn test_energy_arap_spoke() {
    let (tri2vtx, vtx2xyz_ini)
        = crate::trimesh3_primitive::capsule_yup::<f64>(
        0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let (vtx2idx, idx2vtx)
        = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx);
    let (_row2val, idx2val) = {
        let mut row2val = vec!(0f64; num_vtx);
        let mut idx2val = vec!(0f64; idx2vtx.len());
        let mut merge_buffer = vec!(0usize; 0);
        mesh_laplacian_cotangent(
            &tri2vtx, &vtx2xyz_ini,
            &vtx2idx, &idx2vtx,
            &mut row2val, &mut idx2val, &mut merge_buffer);
        (row2val, idx2val)
    };
    let vtx2xyz_def = {
        let mut vtx2xyz_def = vec!(0f64; vtx2xyz_ini.len());
        for i_vtx in 0..num_vtx {
            let x0 = vtx2xyz_ini[i_vtx * 3 + 0];
            let y0 = vtx2xyz_ini[i_vtx * 3 + 1];
            let z0 = vtx2xyz_ini[i_vtx * 3 + 2];
            let x1 = x0 + 0.1 * (3.0 * y0).sin() - 0.1 * (5.0 * z0).cos();
            let y1 = y0 + 0.2 * (4.0 * x0).sin() + 0.2 * (4.0 * z0).cos();
            let z1 = z0 - 0.1 * (5.0 * x0).sin() + 0.1 * (3.0 * y0).cos();
            vtx2xyz_def[i_vtx * 3 + 0] = x1;
            vtx2xyz_def[i_vtx * 3 + 1] = y1;
            vtx2xyz_def[i_vtx * 3 + 2] = z1;
        }
        vtx2xyz_def
    };
    crate::io_obj::save_tri_mesh("target/hoge.obj", &tri2vtx, &vtx2xyz_def);
    for i_vtx in 0..num_vtx {
        let r0 = optimal_rotation_for_arap_spoke(
            i_vtx,
            &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
            &vtx2xyz_ini, &vtx2xyz_def, &idx2val, -1.0);
        let e0 = energy_par_vtx_arap_spoke(
            i_vtx,
            &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
            &vtx2xyz_ini, &vtx2xyz_def, &idx2val, -1.0, &r0);
        let eps = 0.001;
        for i in 0..3 {
            let mut rot = [0f64; 3];
            rot[i] = eps;
            let r1 = r0 * nalgebra::Rotation3::from_euler_angles(rot[0], rot[1], rot[2]);
            let e1 = energy_par_vtx_arap_spoke(
                i_vtx,
                &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
                &vtx2xyz_ini, &vtx2xyz_def, &idx2val, -1.0, &r1);
            assert!(e1 - e0 > 0.);
        }
    }
    let vtx2rot = {
        let mut vtx2rot = vec!(0f64; num_vtx * 9);
        for i_vtx in 0..num_vtx {
            let r0 = optimal_rotation_for_arap_spoke(
                i_vtx,
                &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
                &vtx2xyz_ini, &vtx2xyz_def, &idx2val, -1.0);
            // transpose to change column-major to row-major
            r0.transpose().iter().enumerate().for_each(|(i, &v)| vtx2rot[i_vtx * 9 + i] = v);
        }
        vtx2rot
    };
    let tot_w0 = energy_arap_spoke(
        &vtx2xyz_ini, &vtx2xyz_def,
        &vtx2idx, &idx2vtx, -1., &idx2val, &vtx2rot);
    let eps = 1.0e-5;
    for i_vtx in 0..num_vtx {
        let res = {
            let mut res = nalgebra::Vector3::<f64>::zeros();
            let p0 = del_geo::vec3::to_na(&vtx2xyz_ini, i_vtx);
            let p1 = del_geo::vec3::to_na(&vtx2xyz_def, i_vtx);
            let r_i = nalgebra::Matrix3::<f64>::from_row_slice(&vtx2rot[i_vtx * 9..i_vtx * 9 + 9]);
            for jdx in vtx2idx[i_vtx]..vtx2idx[i_vtx + 1] {
                let j_vtx = idx2vtx[jdx];
                let q0 = del_geo::vec3::to_na(&vtx2xyz_ini, j_vtx);
                let q1 = del_geo::vec3::to_na(&vtx2xyz_def, j_vtx);
                let r_j = nalgebra::Matrix3::<f64>::from_row_slice(&vtx2rot[j_vtx * 9..j_vtx * 9 + 9]);
                let weight = -idx2val[jdx];
                let diff = ((q1 - p1) - (r_i + r_j).scale(0.5) * (q0 - p0)).scale(-4. * weight);
                res += diff;
            }
            res
        };
        for i_dim in 0..3 {
            let vtx2xyz_ptb = {
                let mut vtx2xyz_ptb = vtx2xyz_def.clone();
                vtx2xyz_ptb[i_vtx * 3 + i_dim] += eps;
                vtx2xyz_ptb
            };
            let tot_w1 = energy_arap_spoke(
                &vtx2xyz_ini, &vtx2xyz_ptb,
                &vtx2idx, &idx2vtx, -1., &idx2val, &vtx2rot);
            let dwdp = (tot_w1 - tot_w0) / eps;
            // dbg!(res[i_dim], dwdp);
            assert!((res[i_dim] - dwdp).abs() < dwdp.abs() * 0.001 + 0.0002);
        }
    }
}


pub fn optimal_rotations_mesh_vertx_for_arap_spoke_rim<T>(
    vtx2rot: &mut [T],
    tri2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T])
    where T: num_traits::Float + 'static + nalgebra::RealField + Copy,
          f64: num_traits::AsPrimitive<T>
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    vtx2rot.fill(T::zero());
    for nodes in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (nodes[0], nodes[1], nodes[2]);
        let cots = del_geo::tri3::cot_(
            &vtx2xyz_ini[i0 * 3..i0 * 3 + 3].try_into().unwrap(),
            &vtx2xyz_ini[i1 * 3..i1 * 3 + 3].try_into().unwrap(),
            &vtx2xyz_ini[i2 * 3..i2 * 3 + 3].try_into().unwrap());
        let p0 = del_geo::vec3::to_na(vtx2xyz_ini, i0);
        let p1 = del_geo::vec3::to_na(vtx2xyz_ini, i1);
        let p2 = del_geo::vec3::to_na(vtx2xyz_ini, i2);
        let q0 = del_geo::vec3::to_na(vtx2xyz_def, i0);
        let q1 = del_geo::vec3::to_na(vtx2xyz_def, i1);
        let q2 = del_geo::vec3::to_na(vtx2xyz_def, i2);
        // nalgebra matrix for R^T to make 'vtx2rot' row-major order
        let rt =
            (p1 - p2) * (q1 - q2).transpose().scale(cots[0]) +
                (p2 - p0) * (q2 - q0).transpose().scale(cots[1]) +
                (p0 - p1) * (q0 - q1).transpose().scale(cots[2]);
        vtx2rot[i0 * 9..i0 * 9 + 9].iter_mut().zip(rt.iter()).for_each(|(v, &w)| *v += w);
        vtx2rot[i1 * 9..i1 * 9 + 9].iter_mut().zip(rt.iter()).for_each(|(v, &w)| *v += w);
        vtx2rot[i2 * 9..i2 * 9 + 9].iter_mut().zip(rt.iter()).for_each(|(v, &w)| *v += w);
    }
    for i_vtx in 0..num_vtx {
        let rt = nalgebra::Matrix3::<T>::from_column_slice(&vtx2rot[i_vtx * 9..i_vtx * 9 + 9]);
        let rt = del_geo::mat3::rotational_component(&rt);
        vtx2rot[i_vtx * 9..i_vtx * 9 + 9].iter_mut().zip(rt.iter()).for_each(|(v, &w)| *v += w);
    }
}

fn wdw_arap_spoke_rim<T>(
    p0: &nalgebra::Vector3::<T>,
    p1: &nalgebra::Vector3::<T>,
    p2: &nalgebra::Vector3::<T>,
    q0: &nalgebra::Vector3::<T>,
    q1: &nalgebra::Vector3::<T>,
    q2: &nalgebra::Vector3::<T>,
    rot0: &nalgebra::Matrix3::<T>,
    rot1: &nalgebra::Matrix3::<T>,
    rot2: &nalgebra::Matrix3::<T>) -> (T, [nalgebra::Vector3::<T>; 3])
    where T: nalgebra::RealField + Copy + std::ops::AddAssign + num_traits::Float + 'static,
          f64: num_traits::AsPrimitive<T>
{
    let cots = del_geo::tri3::cot_(p0.as_ref(), p1.as_ref(), p2.as_ref());
    let mut w = T::zero();
    {
        let coeff: T = (0.25f64 / 3.0f64).as_();
        let d12_0 = (q2 - q1) - rot0 * (p2 - p1);
        let d12_1 = (q2 - q1) - rot1 * (p2 - p1);
        let d12_2 = (q2 - q1) - rot2 * (p2 - p1);
        w += coeff * cots[0] * (d12_0.norm_squared() + d12_2.norm_squared() + d12_1.norm_squared());
        //
        let d20_0 = (q0 - q2) - rot0 * (p0 - p2);
        let d20_1 = (q0 - q2) - rot1 * (p0 - p2);
        let d20_2 = (q0 - q2) - rot2 * (p0 - p2);
        w += coeff * cots[1] * (d20_0.norm_squared() + d20_1.norm_squared() + d20_2.norm_squared());
        //
        let d01_0 = (q1 - q0) - rot0 * (p1 - p0);
        let d01_1 = (q1 - q0) - rot1 * (p1 - p0);
        let d01_2 = (q1 - q0) - rot2 * (p1 - p0);
        w += coeff * cots[2] * (d01_0.norm_squared() + d01_1.norm_squared() + d01_2.norm_squared());
    }
    let mut dw = [nalgebra::Vector3::<T>::zeros(); 3];
    {
        let rot = (rot0 + rot1 + rot2).scale(T::one() / 3.0.as_());
        let d12 = (q2 - q1) - rot * (p2 - p1);
        let d20 = (q0 - q2) - rot * (p0 - p2);
        let d01 = (q1 - q0) - rot * (p1 - p0);
        let coeff: T = 0.5f64.as_();
        dw[0] += d20.scale(coeff*cots[1]) - d01.scale(coeff*cots[2]);
        dw[1] += d01.scale(coeff*cots[2]) - d12.scale(coeff*cots[0]);
        dw[2] += d12.scale(coeff*cots[0]) - d20.scale(coeff*cots[1]);
    }
    (w, dw)
}

#[test]
fn test_wdw_arap_spoke_rim() {
    type Vec = nalgebra::Vector3::<f64>;
    type Mat = nalgebra::Matrix3::<f64>;
    let p0 = Vec::new(0., 0., 0.);
    let p1 = Vec::new(1., 2., 3.);
    let p2 = Vec::new(2., 1., 1.);
    let q0 = Vec::new(3., 2., 1.);
    let q1 = Vec::new(3., 0., 4.);
    let q2 = Vec::new(5., 2., 0.);
    let rot0: Mat = nalgebra::Rotation3::from_euler_angles(1., 2., 3.).into();
    let rot1: Mat = nalgebra::Rotation3::from_euler_angles(2., 3., 1.).into();
    let rot2: Mat = nalgebra::Rotation3::from_euler_angles(3., 1., 2.).into();
    let eps = 1.0e-4;
    let (w0, dw0) = wdw_arap_spoke_rim(&p0, &p1, &p2, &q0, &q1, &q2, &rot0, &rot1, &rot2);
    for i_dim in 0..3 {
        let mut q0a: Vec = q0.clone();
        q0a[i_dim] += eps;
        let (w1_0, _dw1_0) = wdw_arap_spoke_rim(&p0, &p1, &p2, &q0a, &q1, &q2, &rot0, &rot1, &rot2);
        assert!(((w1_0 - w0) / eps - dw0[0][i_dim]).abs() < dw0[0][i_dim].abs() * 0.001 + 0.00001);
        //
        let mut q1a: Vec = q1.clone();
        q1a[i_dim] += eps;
        let (w1_1, _dw1_1) = wdw_arap_spoke_rim(&p0, &p1, &p2, &q0, &q1a, &q2, &rot0, &rot1, &rot2);
        assert!(((w1_1 - w0) / eps - dw0[1][i_dim]).abs() < dw0[1][i_dim].abs() * 0.001 + 0.00001);
        //
        let mut q2a: Vec = q2.clone();
        q2a[i_dim] += eps;
        let (w1_2, _dw1_2) = wdw_arap_spoke_rim(&p0, &p1, &p2, &q0, &q1, &q2a, &rot0, &rot1, &rot2);
        assert!(((w1_2-w0)/eps-dw0[2][i_dim]).abs()<dw0[2][i_dim].abs()*0.001+0.00001, );
    }
}


pub fn energy_arap_spoke_rim<T>(
    tri2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    vtx2rot: &[T]) -> T
    where T: nalgebra::RealField + Copy + std::ops::AddAssign + num_traits::Float + 'static,
          f64: num_traits::AsPrimitive<T>
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    let mut tot_w = T::zero();
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz_ini[i0 * 3..i0 * 3 + 3]);
        let p1 = nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz_ini[i1 * 3..i1 * 3 + 3]);
        let p2 = nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz_ini[i2 * 3..i2 * 3 + 3]);
        let q0 = nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz_def[i0 * 3..i0 * 3 + 3]);
        let q1 = nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz_def[i1 * 3..i1 * 3 + 3]);
        let q2 = nalgebra::Vector3::<T>::from_row_slice(&vtx2xyz_def[i2 * 3..i2 * 3 + 3]);
        let (w, _) = wdw_arap_spoke_rim(
            &p0, &p1, &p2,
            &q0, &q1, &q2,
            &nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i0 * 9..i0 * 9 + 9]),
            &nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i1 * 9..i1 * 9 + 9]),
            &nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i2 * 9..i2 * 9 + 9]));
        tot_w += w;
    }
    tot_w
}

#[test]
fn test_energy_arap_spoke_rim() {
    let (tri2vtx, vtx2xyz_ini)
        = crate::trimesh3_primitive::capsule_yup::<f64>(
        0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let vtx2xyz_def = {
        let mut vtx2xyz_def = vec!(0f64; vtx2xyz_ini.len());
        for i_vtx in 0..num_vtx {
            let x0 = vtx2xyz_ini[i_vtx * 3 + 0];
            let y0 = vtx2xyz_ini[i_vtx * 3 + 1];
            let z0 = vtx2xyz_ini[i_vtx * 3 + 2];
            let x1 = x0 + 0.1 * (3.0 * y0).sin() - 0.1 * (5.0 * z0).cos();
            let y1 = y0 + 0.2 * (4.0 * x0).sin() + 0.2 * (4.0 * z0).cos();
            let z1 = z0 - 0.1 * (5.0 * x0).sin() + 0.1 * (3.0 * y0).cos();
            vtx2xyz_def[i_vtx * 3 + 0] = x1;
            vtx2xyz_def[i_vtx * 3 + 1] = y1;
            vtx2xyz_def[i_vtx * 3 + 2] = z1;
        }
        vtx2xyz_def
    };
    crate::io_obj::save_tri_mesh("target/hoge.obj", &tri2vtx, &vtx2xyz_def);
    let mut vtx2rot = vec!(0f64; num_vtx * 9);
    optimal_rotations_mesh_vertx_for_arap_spoke_rim(&mut vtx2rot, &tri2vtx, &vtx2xyz_ini, &vtx2xyz_def);
}

pub fn residual_arap_spoke_rim<T>(
    vtx2res: &mut [T],
    tri2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    vtx2rot: &[T])
    where T: num_traits::Float + 'static + nalgebra::RealField + Copy,
          f64: num_traits::AsPrimitive<T>
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    assert_eq!(vtx2res.len(), num_vtx * 3);
    vtx2res.fill(T::zero());
    for nodes in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (nodes[0], nodes[1], nodes[2]);
        let p0 = del_geo::vec3::to_na(vtx2xyz_ini, i0);
        let p1 = del_geo::vec3::to_na(vtx2xyz_ini, i1);
        let p2 = del_geo::vec3::to_na(vtx2xyz_ini, i2);
        let q0 = del_geo::vec3::to_na(vtx2xyz_def, i0);
        let q1 = del_geo::vec3::to_na(vtx2xyz_def, i1);
        let q2 = del_geo::vec3::to_na(vtx2xyz_def, i2);
        let (_, dw) = wdw_arap_spoke_rim(
            &p0, &p1, &p2,
            &q0, &q1, &q2,
            &nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i0 * 9..i0 * 9 + 9]),
            &nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i1 * 9..i1 * 9 + 9]),
            &nalgebra::Matrix3::<T>::from_row_slice(&vtx2rot[i2 * 9..i2 * 9 + 9]));
        vtx2res[i0 * 3..i0 * 3 + 3].iter_mut().zip(dw[0].iter()).for_each(|(x, y)| *x -= *y);
        vtx2res[i1 * 3..i1 * 3 + 3].iter_mut().zip(dw[1].iter()).for_each(|(x, y)| *x -= *y);
        vtx2res[i2 * 3..i2 * 3 + 3].iter_mut().zip(dw[2].iter()).for_each(|(x, y)| *x -= *y);
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh_laplacian::energy_arap_spoke_rim;

    fn mydef(vtx2xyz_ini: &[f64]) -> Vec<f64> {
        let num_vtx = vtx2xyz_ini.len() / 3;
        let mut vtx2xyz_def = vec!(0f64; vtx2xyz_ini.len());
        for i_vtx in 0..num_vtx {
            let x0 = vtx2xyz_ini[i_vtx * 3 + 0];
            let y0 = vtx2xyz_ini[i_vtx * 3 + 1];
            let z0 = vtx2xyz_ini[i_vtx * 3 + 2];
            let x1 = x0 + 0.1 * (3.0 * y0).sin() - 0.1 * (5.0 * z0).cos();
            let y1 = y0 + 0.2 * (4.0 * x0).sin() + 0.2 * (4.0 * z0).cos();
            let z1 = z0 - 0.1 * (5.0 * x0).sin() + 0.1 * (3.0 * y0).cos();
            vtx2xyz_def[i_vtx * 3 + 0] = x1;
            vtx2xyz_def[i_vtx * 3 + 1] = y1;
            vtx2xyz_def[i_vtx * 3 + 2] = z1;
        }
        vtx2xyz_def
    }

    #[test]
    fn test_energy_arap_spoke_rim_resolution() {
        let (tri2vtx0, vtx2xyz0_ini)
            = crate::trimesh3_primitive::capsule_yup::<f64>(
            0.2, 1.6, 24, 4, 24);
        let vtx2rot0 = {
            let mut vtx2rot0 = vec!(0f64; vtx2xyz0_ini.len() * 3);
            for i in 0..vtx2xyz0_ini.len() / 3 {
                vtx2rot0[i * 9 + 0] = 1.0;
                vtx2rot0[i * 9 + 4] = 1.0;
                vtx2rot0[i * 9 + 8] = 1.0;
            }
            vtx2rot0
        };
        let vtx2xyz0_def = mydef(&vtx2xyz0_ini);
        let w0 = energy_arap_spoke_rim(&tri2vtx0, &vtx2xyz0_ini, &vtx2xyz0_def, &vtx2rot0);
        //
        let (tri2vtx1, vtx2xyz1_ini)
            = crate::trimesh3_primitive::capsule_yup::<f64>(
            0.2, 1.6, 48, 8, 48);
        let vtx2rot1 = {
            let mut vtx2rot1 = vec!(0f64; vtx2xyz1_ini.len() * 3);
            for i in 0..vtx2xyz1_ini.len() / 3 {
                vtx2rot1[i * 9 + 0] = 1.0;
                vtx2rot1[i * 9 + 4] = 1.0;
                vtx2rot1[i * 9 + 8] = 1.0;
            }
            vtx2rot1
        };
        let vtx2xyz1_def = mydef(&vtx2xyz1_ini);
        let w1 = energy_arap_spoke_rim(&tri2vtx1, &vtx2xyz1_ini, &vtx2xyz1_def, &vtx2rot1);
        //
        let (tri2vtx2, vtx2xyz2_ini)
            = crate::trimesh3_primitive::capsule_yup::<f64>(
            0.2, 1.6, 96, 16, 96);
        let vtx2rot2 = {
            let mut vtx2rot2 = vec!(0f64; vtx2xyz2_ini.len() * 3);
            for i in 0..vtx2xyz2_ini.len() / 3 {
                vtx2rot2[i * 9 + 0] = 1.0;
                vtx2rot2[i * 9 + 4] = 1.0;
                vtx2rot2[i * 9 + 8] = 1.0;
            }
            vtx2rot2
        };
        let vtx2xyz2_def = mydef(&vtx2xyz2_ini);
        let w2 = energy_arap_spoke_rim(&tri2vtx2, &vtx2xyz2_ini, &vtx2xyz2_def, &vtx2rot2);
        assert!((w0-w1).abs()<w1*0.01);
        assert!((w1-w2).abs()<w2*0.004);
    }
}