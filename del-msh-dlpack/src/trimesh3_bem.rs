use del_dlpack::{
    check_2d_tensor as chk2, dlpack, get_managed_tensor_from_pyany as get_tensor,
    get_shape_tensor as shape, slice, slice_mut,
};
use pyo3::{prelude::PyModule, pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(
        trimesh3_bem_eval_kelvinlet_bruteforce,
        m
    )?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn trimesh3_bem_eval_kelvinlet_bruteforce(
    _py: Python<'_>,
    tri2vtx: &Bound<'_, PyAny>,
    vtx2xyz: &Bound<'_, PyAny>,
    vtx2rhs: &Bound<'_, PyAny>,
    wtx2xyz: &Bound<'_, PyAny>,
    wtx2lhs: &Bound<'_, PyAny>,
    mu: f32,
    nu: f32,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let tri2vtx = get_tensor(tri2vtx)?;
    let vtx2xyz = get_tensor(vtx2xyz)?;
    let vtx2rhs = get_tensor(vtx2rhs)?;
    let wtx2xyz = get_tensor(wtx2xyz)?;
    let wtx2lhs = get_tensor(wtx2lhs)?;
    let device_type = tri2vtx.ctx.device_type;
    let num_tri = shape(tri2vtx, 0).unwrap();
    let num_vtx = shape(vtx2xyz, 0).unwrap();
    let num_wtx = shape(wtx2xyz, 0).unwrap();
    chk2::<u32>(tri2vtx, num_tri, 3, device_type).unwrap();
    chk2::<f32>(vtx2xyz, num_vtx, 3, device_type).unwrap();
    chk2::<f32>(vtx2rhs, num_vtx, 3, device_type).unwrap();
    chk2::<f32>(wtx2xyz, num_wtx, 3, device_type).unwrap();
    chk2::<f32>(wtx2lhs, num_wtx, 3, device_type).unwrap();
    match device_type {
        dlpack::device_type_codes::CPU => {
            let tri2vtx_u32 = slice!(tri2vtx, u32).unwrap().as_chunks::<3>().0;
            let tri2vtx_usize: Vec<[usize; 3]> = tri2vtx_u32
                .iter()
                .map(|&[a, b, c]| [a as usize, b as usize, c as usize])
                .collect();
            del_msh_cpu::trimesh3_bem::eval_kelvinlet_bruteforce(
                &tri2vtx_usize,
                slice!(vtx2xyz, f32).unwrap().as_chunks::<3>().0,
                slice!(vtx2rhs, f32).unwrap().as_chunks::<3>().0,
                slice!(wtx2xyz, f32).unwrap().as_chunks::<3>().0,
                slice_mut!(wtx2lhs, f32).unwrap().as_chunks_mut::<3>().0,
                mu,
                nu,
            );
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "only CPU is supported",
            ));
        }
    }
    Ok(())
}
