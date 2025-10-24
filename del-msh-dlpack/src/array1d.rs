use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

pub fn add_functions(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(array1d_permute, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(array1d_argsort, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn array1d_permute(
    _py: Python<'_>,
    old2val: &Bound<'_, PyAny>,
    new2old: &Bound<'_, PyAny>,
    new2val: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let old2val = crate::get_managed_tensor_from_pyany(old2val)?;
    let new2old = crate::get_managed_tensor_from_pyany(new2old)?;
    let new2val = crate::get_managed_tensor_from_pyany(new2val)?;
    let n = crate::get_shape_tensor(old2val, 0);
    let device = old2val.ctx.device_type;
    //
    crate::check_1d_tensor::<u32>(old2val, n, device).unwrap();
    crate::check_1d_tensor::<u32>(new2old, n, device).unwrap();
    crate::check_1d_tensor::<u32>(new2val, n, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let old2val = unsafe { crate::slice_from_tensor::<u32>(old2val) }.unwrap();
            let new2old = unsafe { crate::slice_from_tensor::<u32>(new2old) }.unwrap();
            let new2val = unsafe { crate::slice_from_tensor_mut::<u32>(new2val) }.unwrap();
            for i_new in 0..n as usize {
                new2val[i_new] = old2val[new2old[i_new] as usize];
            }
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            let old2val = del_cudarc_sys::CuVec::<u32>::from_dptr(
                old2val.data as cu::CUdeviceptr,
                n as usize,
            );
            let new2old = del_cudarc_sys::CuVec::<u32>::from_dptr(
                new2old.data as cu::CUdeviceptr,
                n as usize,
            );
            let new2val = del_cudarc_sys::CuVec::<u32>::from_dptr(
                new2val.data as cu::CUdeviceptr,
                n as usize,
            );
            del_cudarc_sys::util::permute(stream, &new2val, &new2old, &old2val).unwrap();
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

fn sort_by_key(data: &mut [u32], v: &mut [u32]) {
    let len = data.len();
    fn quicksort(d: &mut [u32], lo: usize, hi: usize, v: &mut [u32]) {
        if lo >= hi {
            return;
        }
        let mut i = lo;
        for j in lo..hi {
            if d[j] < d[hi] {
                d.swap(i, j);
                v.swap(i, j);
                i += 1;
            }
        }
        d.swap(i, hi);
        v.swap(i, hi);
        if i > 0 {
            quicksort(d, lo, i - 1, v);
        }
        quicksort(d, i + 1, hi, v);
    }
    if len > 1 {
        quicksort(data, 0, len - 1, v);
    }
}

#[pyo3::pyfunction]
pub fn array1d_argsort(
    _py: Python<'_>,
    idx2val: &Bound<'_, PyAny>,
    jdx2idx: &Bound<'_, PyAny>,
    #[allow(unused_variables)] stream_ptr: u64,
) -> PyResult<()> {
    let idx2val = crate::get_managed_tensor_from_pyany(idx2val)?;
    let jdx2idx = crate::get_managed_tensor_from_pyany(jdx2idx)?;
    //
    let n = crate::get_shape_tensor(jdx2idx, 0);
    let device = jdx2idx.ctx.device_type;
    //
    crate::check_1d_tensor::<u32>(jdx2idx, n, device).unwrap();
    crate::check_1d_tensor::<u32>(idx2val, n, device).unwrap();
    //
    match device {
        dlpack::device_type_codes::CPU => {
            let idx2val = unsafe { crate::slice_from_tensor_mut::<u32>(idx2val) }.unwrap();
            let jdx2idx = unsafe { crate::slice_from_tensor_mut::<u32>(jdx2idx) }.unwrap();
            jdx2idx
                .iter_mut()
                .enumerate()
                .for_each(|(iv, idx)| *idx = iv as u32);
            // jdx2idx.sort_by_key(|iv| idx2val[*iv as usize]);
            sort_by_key(idx2val, jdx2idx);
        }
        #[cfg(feature = "cuda")]
        dlpack::device_type_codes::GPU => {
            use del_cudarc_sys::cu;
            use del_cudarc_sys::cuda_check;
            cuda_check!(cu::cuInit(0));
            let stream = del_cudarc_sys::stream_from_u64(stream_ptr);
            //
            let jdx2idx = del_cudarc_sys::CuVec::<u32>::from_dptr(
                jdx2idx.data as cu::CUdeviceptr,
                n as usize,
            );
            del_cudarc_sys::util::set_consecutive_sequence(stream, &jdx2idx);
            let idx2val = del_cudarc_sys::CuVec::<u32>::from_dptr(
                idx2val.data as cu::CUdeviceptr,
                n as usize,
            );
            del_cudarc_sys::sort_by_key_u32::radix_sort_by_key_u32(stream, &idx2val, &jdx2idx)
                .unwrap();
        }
        _ => {
            panic!()
        }
    }
    Ok(())
}
