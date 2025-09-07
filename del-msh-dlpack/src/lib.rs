use dlpack::ManagedTensor;
use pyo3::prelude::{PyAnyMethods, PyCapsuleMethods};
use pyo3::types::PyCapsule;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};
use std::{mem, slice};

mod edge2vtx;
mod trimesh3_raycast;

#[pyo3::pymodule]
#[pyo3(name = "del_msh_dlpack")]
fn del_msh_dlpack_(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(solve, m)?)?;
    trimesh3_raycast::add_functions(_py, m)?;
    edge2vtx::add_functions(_py, m)?;
    Ok(())
}

// --------------------------------

pub fn get_managed_tensor_from_pyany<'a>(
    vtx2idx: &'a pyo3::Bound<'a, PyAny>,
) -> PyResult<&'a dlpack::Tensor> {
    let capsule = vtx2idx.downcast::<PyCapsule>()?;
    let ptr = capsule.pointer() as *mut ManagedTensor;
    if ptr.is_null() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Null ManagedTensor",
        ));
    }
    let tensor = unsafe { &(*ptr).dl_tensor };
    Ok(tensor)
}

///
/// # Safety
pub unsafe fn is_c_contiguous(shape: *const i64, strides: *const i64, ndim: i32) -> bool {
    // スカラーや 0/1 長次元のみ → 連続扱い（strides が何でも OK）
    if ndim <= 0 {
        return true;
    }
    if strides.is_null() {
        // DLPack では strides == NULL は「コンパクトな C 連続」を意味
        return true;
    }

    let n = ndim as usize;
    let sh = slice::from_raw_parts(shape, n);
    let st = slice::from_raw_parts(strides, n);

    // 後ろの次元から expected=1 で積み上げる（dim<=1 は無視）
    let mut expected: i64 = 1;
    for i in (0..n).rev() {
        let dim = sh[i];
        if dim > 1 {
            // 負の stride（逆順ビュー）は非連続扱いにする
            if st[i] != expected {
                return false;
            }
            // 次の expected を更新
            expected = expected.saturating_mul(dim);
        }
    }
    true
}

///
/// # Safety
pub unsafe fn slice_shape_from_tensor<'a, T>(t: &dlpack::Tensor) -> Option<(&'a [T], Vec<i64>)> {
    assert!(is_c_contiguous(t.shape, t.strides, t.ndim));
    if t.shape.is_null() || t.ndim < 0 {
        return None;
    }
    let sh = slice::from_raw_parts(t.shape, t.ndim as usize); // ← ここでスライス化
    let len = sh
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))?;
    let s = if t.data.is_null() {
        assert_eq!(len, 0, "null なら len==0 である必要があります");
        &[]
    } else {
        // アラインメント自己診断（本番では assert を外してもOK）
        assert_eq!(
            (t.data as usize) % mem::align_of::<T>(),
            0,
            "unaligned pointer for T"
        );
        slice::from_raw_parts(t.data as *const T, len)
    };
    Some((s, sh.to_vec()))
}

///
/// # Safety
pub unsafe fn slice_shape_from_tensor_mut<'a, T>(
    t: &dlpack::Tensor,
) -> Option<(&'a mut [T], Vec<i64>)> {
    assert!(is_c_contiguous(t.shape, t.strides, t.ndim));
    if t.shape.is_null() || t.ndim < 0 {
        return None;
    }
    let sh = slice::from_raw_parts(t.shape, t.ndim as usize); // ← ここでスライス化
    let len = sh
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))?;
    let s = if t.data.is_null() {
        assert_eq!(len, 0, "null なら len==0 である必要があります");
        slice::from_raw_parts_mut(std::ptr::NonNull::<T>::dangling().as_ptr(), 0)
    } else {
        // アラインメント自己診断（本番では assert を外してもOK）
        assert_eq!(
            (t.data as usize) % mem::align_of::<T>(),
            0,
            "unaligned pointer for T"
        );
        slice::from_raw_parts_mut(t.data as *mut T, len)
    };
    Some((s, sh.to_vec()))
}
