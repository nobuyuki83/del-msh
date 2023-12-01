use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn from_grid<T>(
    nx: usize,
    ny: usize) -> (Vec<usize>, Vec<T>)
    where T: num_traits::Float + 'static,
          f32: num_traits::AsPrimitive<T>,
          usize: num_traits::AsPrimitive<T>
{
    let np = (nx + 1) * (ny + 1);
    let mut vtx2xy: Vec<T> = vec![0_f32.as_(); np * 2];
    for iy in 0..ny + 1 {
        for ix in 0..nx + 1 {
            let ip = iy * (nx + 1) + ix;
            vtx2xy[ip * 2 + 0] = ix.as_();
            vtx2xy[ip * 2 + 1] = iy.as_();
        }
    }
    let mut quad2vtx = vec![0; nx * ny * 4];
    for iy in 0..ny {
        for ix in 0..nx {
            let iq = iy * nx + ix;
            quad2vtx[iq * 4 + 0] = (iy + 0) * (nx + 1) + (ix + 0);
            quad2vtx[iq * 4 + 1] = (iy + 0) * (nx + 1) + (ix + 1);
            quad2vtx[iq * 4 + 2] = (iy + 1) * (nx + 1) + (ix + 1);
            quad2vtx[iq * 4 + 3] = (iy + 1) * (nx + 1) + (ix + 0);
        }
    }
    (quad2vtx, vtx2xy)
}

#[test]
fn test_grid_quad2() {
    from_grid::<f32>(12, 5);
    from_grid::<f64>(12, 5);
}