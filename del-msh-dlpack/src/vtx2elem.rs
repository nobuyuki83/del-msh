use crate::{check_2d_tensor, get_shape_tensor};
use pyo3::{pyfunction, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<pyo3::types::PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(vtx2elem_from_uniform_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn vtx2elem_from_uniform_mesh(
    py: Python<'_>,
    elem2vtx: &Bound<'_, PyAny>,
    num_vtx: usize,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<(pyo3::Py<PyAny>, pyo3::Py<PyAny>)> {
    let elem2vtx = crate::get_managed_tensor_from_pyany(elem2vtx)?;
    //
    let num_elem = get_shape_tensor(elem2vtx, 0);
    let num_node = get_shape_tensor(elem2vtx, 1);
    let device = elem2vtx.ctx.device_type;
    check_2d_tensor::<u32>(elem2vtx, num_elem, num_node, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let elem2vtx = unsafe { crate::slice_from_tensor::<u32>(elem2vtx).unwrap() };
            let (vtx2idx, idx2elem) =
                del_msh_cpu::vtx2elem::from_uniform_mesh(elem2vtx, num_node as usize, num_vtx);
            let vtx2idx_cap = crate::make_capsule_from_vec(py, vec![vtx2idx.len() as i64], vtx2idx);
            let idx2elem_cap =
                crate::make_capsule_from_vec(py, vec![idx2elem.len() as i64], idx2elem);
            Ok((vtx2idx_cap, idx2elem_cap))
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::{cu, cuda_check, CuVec};
            cuda_check!(cu::cuInit(0)).unwrap();
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            use del_cudarc_sys::cu::CUdeviceptr;
            let elem2vtx = CuVec::<u32>::new(
                elem2vtx.data as CUdeviceptr,
                (num_elem * num_node) as usize,
                false,
            );
            let (vtx2idx, idx2elem) = del_msh_cudarc_sys::vtx2elem::from_uniform_mesh(
                stream,
                &elem2vtx,
                num_elem as usize,
                num_vtx,
            );
            let vtx2idx_cap =
                crate::make_capsule_from_cuvec(py, 0, vec![vtx2idx.n as i64], vtx2idx);
            let idx2elem_cap =
                crate::make_capsule_from_cuvec(py, 0, vec![idx2elem.n as i64], idx2elem);
            Ok((vtx2idx_cap, idx2elem_cap))
        }
        _ => {
            todo!()
        }
    }
}
