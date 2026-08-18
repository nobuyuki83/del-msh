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
