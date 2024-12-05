use candle_core::{Tensor, Var};

pub struct Optimizer {
    vtx2xyz: Var,
    ls: del_fem_ls::linearsystem::Solver<f32>,
    pub tri2vtx: Tensor,
    pub lambda: f64,
    pub learning_rate: f64,
}

impl Optimizer {
    pub fn new(
        var: Var,
        learning_rate: f64,
        tri2vtx: Tensor,
        num_vtx: usize,
        lambda: f64,
    ) -> candle_core::Result<Self> {
        let ls = {
            let tri2vtx: Vec<usize> = tri2vtx
                .flatten_all()?
                .to_vec1::<u32>()?
                .iter()
                .map(|v| *v as usize)
                .collect();
            del_fem_core::laplace_tri3::to_linearsystem(&tri2vtx, num_vtx, 1., lambda as f32)
        };
        let adm = Optimizer {
            vtx2xyz: var,
            learning_rate,
            ls,
            tri2vtx,
            lambda,
        };
        Ok(adm)
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
        if let Some(dw_vtx2xyz) = grads.get(&self.vtx2xyz) {
            let num_vtx = dw_vtx2xyz.dims2()?.0;
            let grad = {
                self.ls.r_vec = dw_vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
                self.ls.solve_cg();
                Tensor::from_vec(
                    self.ls.u_vec.clone(),
                    (num_vtx, 3),
                    &candle_core::Device::Cpu,
                )?
            };
            let delta = (grad * self.learning_rate)?; // gradient descent
                                                      /*
                                                      let delta = {
                                                          self.ls.r_vec = delta.flatten_all()?.to_vec1::<f32>()?;
                                                          self.ls.solve_cg();
                                                          Tensor::from_vec(self.ls.u_vec.clone(), (num_vtx, 3), &candle_core::Device::Cpu)?
                                                      };
                                                       */
            self.vtx2xyz.set(&self.vtx2xyz.sub(&(delta))?)?;
        }
        Ok(())
    }
}
