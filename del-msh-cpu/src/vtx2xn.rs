//! functions related to c-style contiguous array of N dimensional coordinates

use num_traits::AsPrimitive;

pub fn cast<T, U>(vtx2xyz0: &[U]) -> Vec<T>
where
    T: Copy + 'static,
    U: AsPrimitive<T>,
{
    let res: Vec<T> = vtx2xyz0.iter().map(|v| v.as_()).collect();
    res
}

pub fn cog<T, const N: usize>(vtx2xyz: &[[T; N]]) -> [T; N]
where
    T: num_traits::Float + Copy + 'static + std::iter::Sum<T>,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vecn::VecN;
    let mut cog = [T::zero(); N];
    for xyz in vtx2xyz.iter() {
        cog.add_in_place(xyz);
    }
    let s = T::one() / vtx2xyz.len().as_();
    cog.scale_in_place(s);
    cog
}

pub fn cov_cog<T, const N: usize>(vtx2xyz: &[[T; N]]) -> ([[T; N]; N], [T; N])
where
    T: num_traits::Float + Copy + 'static + std::iter::Sum,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vecn::VecN;
    let cog = cog::<T, N>(vtx2xyz);
    let mut cov = [[T::zero(); N]; N];
    for xyz in vtx2xyz.iter() {
        let d = xyz.sub(&cog);
        for i in 0..N {
            for j in 0..N {
                cov[i][j] = cov[i][j] + d[i] * d[j];
            }
        }
    }
    (cov, cog)
}

pub fn set_zero<T, const N: usize>(p: &mut [[T; N]])
where
    T: num_traits::Float,
{
    p.iter_mut().for_each(|v| *v = [T::zero(); N]);
}

pub fn dot<T, const N: usize>(v0: &[[T; N]], v1: &[[T; N]]) -> T
where
    T: num_traits::Float,
{
    assert_eq!(v0.len(), v1.len());
    v0.iter().zip(v1.iter()).fold(T::zero(), |sum, (&x, &y)| {
        sum + del_geo_core::vecn::dot(&x, &y)
    })
}

pub fn copy<T, const N: usize>(p: &mut [[T; N]], u: &[[T; N]])
where
    T: Copy,
{
    assert_eq!(p.len(), u.len());
    p.iter_mut().zip(u.iter()).for_each(|(a, &b)| *a = b);
}

pub fn add_scaled_vector<T, const N: usize>(u: &mut [[T; N]], alpha: T, p: &[[T; N]])
where
    T: num_traits::Float,
{
    use del_geo_core::vecn::VecN;
    assert_eq!(u.len(), p.len());
    u.iter_mut()
        .zip(p.iter())
        .for_each(|(a, &b)| (*a).add_in_place(&b.scale(alpha)));
}

/// {p} = {r} + beta*{p}
pub fn scale_and_add_vec<T, const N: usize>(p: &mut [[T; N]], beta: T, r: &[[T; N]])
where
    T: num_traits::Float,
{
    use del_geo_core::vecn::VecN;
    assert_eq!(r.len(), p.len());
    for i in 0..p.len() {
        p[i] = r[i].add(&p[i].scale(beta));
    }
}

pub fn set_fixed<T, const NDIMVAL: usize>(
    blk2val: &mut [[T; NDIMVAL]],
    blk2isfix: &[[i32; NDIMVAL]],
) where
    T: num_traits::Float,
{
    assert_eq!(blk2val.len(), blk2isfix.len());
    for i_blk in 0..blk2val.len() {
        for i_dimval in 0..NDIMVAL {
            if blk2isfix[i_blk][i_dimval] == 0 {
                continue;
            }
            blk2val[i_blk][i_dimval] = T::zero();
        }
    }
}
