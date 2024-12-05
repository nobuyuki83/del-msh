use candle_core::Device::Cpu;
use candle_core::Tensor;

pub fn peturb_2d_tensor(
    t: &Tensor,
    i_row: usize,
    i_col: usize,
    delta: f64,
) -> candle_core::Result<Tensor> {
    assert!(t.dtype().is_float());
    let num_row = t.dims2()?.0;
    let num_col = t.dims2()?.1;
    match t.dtype() {
        candle_core::DType::F32 => {
            let mut v0 = t.flatten_all()?.to_vec1::<f32>()?;
            v0[i_row * num_col + i_col] += delta as f32;
            Tensor::from_vec(v0, (num_row, num_col), &Cpu)
        }
        _ => {
            todo!()
        }
    }
}
