/// TODO: it might be better to specify the normal vector
pub fn to_trimesh3_torus(
    vtx2xyz: &nalgebra::Matrix3xX<f32>,
    vtx2bin: &nalgebra::Matrix3xX<f32>,
    rad: f32,
    ndiv_circum: usize,
) -> (Vec<usize>, Vec<f32>) {
    let n = ndiv_circum;
    let dtheta = std::f32::consts::PI * 2. / n as f32;
    let num_vtx = vtx2xyz.ncols();
    let mut pnt2xyz = Vec::<f32>::new();
    for ipnt in 0..num_vtx {
        let p0 = vtx2xyz.column(ipnt).into_owned();
        let p1 = vtx2xyz.column((ipnt + 1) % num_vtx).into_owned();
        let z0 = (p1 - p0).normalize();
        let x0 = vtx2bin.column(ipnt);
        let y0 = z0.cross(&x0);
        for i in 0..n {
            let theta = dtheta * i as f32;
            let v0 = x0.scale(theta.cos()) + y0.scale(theta.sin());
            let q0 = p0 + v0.scale(rad);
            q0.iter().for_each(|&v| pnt2xyz.push(v));
        }
    }

    let mut tri2pnt = Vec::<usize>::new();
    for iseg in 0..num_vtx {
        let ipnt0 = iseg;
        let ipnt1 = (ipnt0 + 1) % num_vtx;
        for i in 0..n {
            tri2pnt.push(ipnt0 * n + i);
            tri2pnt.push(ipnt0 * n + (i + 1) % n);
            tri2pnt.push(ipnt1 * n + i);
            //
            tri2pnt.push(ipnt1 * n + (i + 1) % n);
            tri2pnt.push(ipnt1 * n + i);
            tri2pnt.push(ipnt0 * n + (i + 1) % n);
        }
    }
    (tri2pnt, pnt2xyz)
}
