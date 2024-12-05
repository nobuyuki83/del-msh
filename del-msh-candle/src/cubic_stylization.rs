fn rotate90(edge2xy: candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let x = edge2xy.get_on_dim(1, 0)?;
    let y = edge2xy.get_on_dim(1, 1)?;
    candle_core::Tensor::stack(&[(y * -1.)?, x], 1)
}

pub fn loss(
    vtx2xy: &candle_core::Tensor,
    edge2vtx: &[usize],
) -> candle_core::Result<candle_core::Tensor> {
    let vtx2xyz_to_edgevector = crate::vtx2xyz_to_edgevector::Layer {
        edge2vtx: Vec::<usize>::from(edge2vtx),
    };
    let edge2xy = vtx2xy.apply_op1(vtx2xyz_to_edgevector)?;
    let edge2nrm = rotate90(edge2xy)?;
    let mut edge2norm_trg = edge2nrm.flatten_all()?.to_vec1::<f32>()?;
    for norm in edge2norm_trg.chunks_mut(2) {
        let x0 = norm[0];
        let y0 = norm[1];
        let len = (x0 * x0 + y0 * y0).sqrt();
        if y0 > x0 && y0 > -x0 {
            norm[0] = 0f32;
            norm[1] = len;
        }
        if y0 < x0 && y0 < -x0 {
            norm[0] = 0f32;
            norm[1] = -len;
        }
        if y0 > x0 && y0 < -x0 {
            norm[0] = -len;
            norm[1] = 0f32;
        }
        if y0 < x0 && y0 > -x0 {
            norm[0] = len;
            norm[1] = 0f32;
        }
    }
    let edge2norm_trg = candle_core::Tensor::from_slice(
        edge2norm_trg.as_slice(),
        candle_core::Shape::from((edge2norm_trg.len() / 2, 2)),
        &candle_core::Device::Cpu,
    )?;
    let unorm_diff = edge2nrm.sub(&edge2norm_trg)?.sqr()?.sum_all()?;
    Ok(unorm_diff)
}
