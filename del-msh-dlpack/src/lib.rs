use dlpack::{DataType, ManagedTensor, Tensor};
use pyo3::prelude::{PyAnyMethods, PyCapsuleMethods};
use pyo3::types::PyCapsule;
use pyo3::{types::PyModule, Bound, PyAny, PyResult, Python};

mod array1d;
mod edge2vtx;
mod mortons;
mod quad_oct_tree;
mod trimesh3;
mod trimesh3_raycast;
mod vtx2elem;
mod vtx2vtx;

#[pyo3::pymodule]
#[pyo3(name = "del_msh_dlpack")]
fn del_msh_dlpack_(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(solve, m)?)?;
    trimesh3_raycast::add_functions(_py, m)?;
    trimesh3::add_functions(_py, m)?;
    edge2vtx::add_functions(_py, m)?;
    vtx2vtx::add_functions(_py, m)?;
    vtx2elem::add_functions(_py, m)?;
    mortons::add_functions(_py, m)?;
    array1d::add_functions(_py, m)?;
    quad_oct_tree::add_functions(_py, m)?;
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
    use std::slice;
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
pub unsafe fn is_tensor_c_contiguous(t: &Tensor) -> bool {
    is_c_contiguous(t.shape, t.strides, t.ndim)
}

///
/// # Safety
pub unsafe fn slice_from_tensor<'a, T>(t: &dlpack::Tensor) -> Option<&'a [T]> {
    use std::{mem, slice};
    assert_eq!(mem::size_of::<T>() * 8, t.dtype.bits as usize);
    assert!(
        is_c_contiguous(t.shape, t.strides, t.ndim),
        "not contiguous"
    );
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
    Some(s)
}

///
/// # Safety
pub unsafe fn slice_from_tensor_mut<'a, T>(t: &dlpack::Tensor) -> Option<&'a mut [T]> {
    assert!(is_c_contiguous(t.shape, t.strides, t.ndim));
    if t.shape.is_null() || t.ndim < 0 {
        return None;
    }
    use std::slice::{from_raw_parts, from_raw_parts_mut};
    let sh = from_raw_parts(t.shape, t.ndim as usize); // ← ここでスライス化
    let len = sh
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))?;
    let s = if t.data.is_null() {
        assert_eq!(len, 0, "null なら len==0 である必要があります");
        from_raw_parts_mut(std::ptr::NonNull::<T>::dangling().as_ptr(), 0)
    } else {
        // アラインメント自己診断（本番では assert を外してもOK）
        assert_eq!(
            (t.data as usize) % std::mem::align_of::<T>(),
            0,
            "unaligned pointer for T"
        );
        from_raw_parts_mut(t.data as *mut T, len)
    };
    Some(s)
}

// -------------------------------------------------

pub trait ToDataTypeCode {
    fn category() -> dlpack::DataTypeCode;
}

impl ToDataTypeCode for u32 {
    fn category() -> dlpack::DataTypeCode {
        dlpack::data_type_codes::UINT
    }
}

impl ToDataTypeCode for usize {
    fn category() -> dlpack::DataTypeCode {
        dlpack::data_type_codes::UINT
    }
}

impl ToDataTypeCode for i32 {
    fn category() -> dlpack::DataTypeCode {
        dlpack::data_type_codes::INT
    }
}

impl ToDataTypeCode for u64 {
    fn category() -> dlpack::DataTypeCode {
        dlpack::data_type_codes::UINT
    }
}

impl ToDataTypeCode for f32 {
    fn category() -> dlpack::DataTypeCode {
        dlpack::data_type_codes::FLOAT
    }
}

// ----------------------------------------

/// Rust側の所有物をまとめておき、deleterで回収する用
#[repr(C)]
struct DlpackContextVec<T> {
    data_owner: *mut Vec<T>, // 実データ
    shape_owner: *mut [i64], // 形状配列（Box<[i64]>）
                             // strides を持たせたい場合は *mut [i64] を追加
}

extern "C" fn dl_managed_tensor_deleter_vec<T>(m: *mut dlpack::ManagedTensor) {
    if m.is_null() {
        return;
    }
    unsafe {
        // manager_ctx を回収
        let ctx = (*m).manager_ctx as *mut DlpackContextVec<T>;
        if !ctx.is_null() {
            let ctx_box = Box::from_raw(ctx);
            if !ctx_box.data_owner.is_null() {
                let _ = Box::from_raw(ctx_box.data_owner); // drop
            }
            if !ctx_box.shape_owner.is_null() {
                let _ = Box::from_raw(ctx_box.shape_owner); // drop
            }
        }
        let _ = Box::from_raw(m); // DLManagedTensor 自体を解放
    }
}

/// PyCapsule のデストラクタ
/// - まだ消費されていない（名前が "dltensor"）場合のみ、DLManagedTensor.deleter を呼ぶ
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name = std::ffi::CString::new("dltensor").unwrap();
    let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr());
    if ptr.is_null() {
        pyo3::ffi::PyErr_Clear(); // ← これが肝心（例外を外へ漏らさない）
        return;
    }
    let m = ptr as *mut dlpack::ManagedTensor;
    if !m.is_null() {
        let deleter = (*m).deleter;
        deleter(m);
    }
}

fn make_capsule_from_vec<T>(py: Python, shape_vec: Vec<i64>, mut data: Vec<T>) -> pyo3::Py<PyAny>
where
    T: ToDataTypeCode,
{
    // 生ポインタ
    let data_ptr = data.as_mut_ptr() as *mut std::os::raw::c_void;
    let ndim = shape_vec.len();

    // shape を Box<[i64]> にして保持
    let shape_boxed: Box<[i64]> = shape_vec.into_boxed_slice();
    let shape_ptr = shape_boxed.as_ptr() as *mut i64;

    // 所有権を raw に移しておく（deleter で回収）
    let data_owner = Box::into_raw(Box::new(data));
    let shape_owner = Box::into_raw(shape_boxed);

    // 2) manager_ctx
    let ctx = DlpackContextVec {
        data_owner,
        shape_owner,
    };
    let ctx_ptr = Box::into_raw(Box::new(ctx)) as *mut std::os::raw::c_void;

    let dl_tensor = dlpack::Tensor {
        data: data_ptr,
        ctx: dlpack::Context {
            device_type: dlpack::device_type_codes::CPU,
            device_id: 0,
        },
        ndim: ndim as i32,
        dtype: dlpack::DataType {
            code: T::category(), // f32
            bits: (std::mem::size_of::<T>() * 8) as u8,
            lanes: 1,
        },
        shape: shape_ptr,
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };

    // 3) DLManagedTensor を構築（C連続: strides=NULL）
    let managed = Box::new(dlpack::ManagedTensor {
        manager_ctx: ctx_ptr,
        deleter: dl_managed_tensor_deleter_vec::<T>,
        dl_tensor,
    });

    // 4) PyCapsule("dltensor") を作成（消費側に所有権移譲）
    let raw_managed = Box::into_raw(managed);
    // let name = std::ffi::CString::new("dltensor").unwrap();
    let name_ptr: *const std::ffi::c_char = c"dltensor".as_ptr() as *const std::ffi::c_char;

    unsafe {
        let cap_ptr = pyo3::ffi::PyCapsule_New(
            raw_managed as *mut std::os::raw::c_void,
            // name.as_ptr() as *const std::os::raw::c_char,
            name_ptr,
            Some(capsule_destructor),
        );
        pyo3::Py::<PyAny>::from_owned_ptr(py, cap_ptr)
    }
}

// -------------------------------

#[allow(dead_code)]
extern "C" fn dlpack_deleter_wo_free(mt: *mut dlpack::ManagedTensor) {
    unsafe {
        if mt.is_null() {
            return;
        }
        let m = &mut *mt;
        if !m.dl_tensor.shape.is_null() {
            // shape と strides を Box 回収（strides は null の可能性）
            let _ = Box::from_raw(m.dl_tensor.shape);
        }
        if !m.dl_tensor.strides.is_null() {
            let _ = Box::from_raw(m.dl_tensor.strides);
        }
        // data（CUdeviceptr）は**解放しない**（所有権なし）
        let _ = Box::from_raw(mt);
    }
}

#[cfg(feature = "cuda")]
fn make_capsule_from_cuvec<T: ToDataTypeCode>(
    py: Python,
    device_id: i32,
    shape: Vec<i64>,
    mut v: del_cudarc_sys::CuVec<T>,
) -> pyo3::Py<PyAny> {
    v.is_free_at_drop = false;

    // --- 必要ならここで「可視化」 ---
    // 例: 既に他ストリームで書き込み済みで、from_dlpack 側の消費前に可視化したい場合
    // unsafe { del_cudarc_sys::cuStreamSynchronize(stream); }
    // あるいは __dlpack__ プロトコルを実装して consumer stream を受ける設計にする（高度）

    // shape/strides をヒープ確保（PyTorch が later に deleter を呼ぶ前提）
    let ndim = shape.len() as i32;
    let shape_box = shape.into_boxed_slice();
    let shape_ptr = Box::into_raw(shape_box) as *mut i64;

    // 連続メモリなら strides=null
    let strides_ptr: *mut i64 = std::ptr::null_mut();

    // data ポインタ（注意: CUdeviceptr は u64。*mut c_void にキャスト）
    let data_ptr = v.dptr as usize as *mut std::ffi::c_void;

    let dl = Tensor {
        data: data_ptr,
        ctx: dlpack::Context {
            device_type: dlpack::device_type_codes::GPU,
            device_id,
        },
        ndim,
        dtype: dlpack::DataType {
            code: T::category(),
            bits: size_of::<T>() as u8 * 8,
            lanes: 1,
        },
        shape: shape_ptr,
        strides: strides_ptr,
        byte_offset: 0,
    };

    let managed = Box::new(dlpack::ManagedTensor {
        dl_tensor: dl,
        manager_ctx: std::ptr::null_mut(),
        deleter: dlpack_deleter_wo_free,
    });

    // 4) PyCapsule("dltensor") を作成（消費側に所有権移譲）
    let raw_managed = Box::into_raw(managed);
    // let name = std::ffi::CString::new("dltensor").unwrap();
    let name_ptr: *const std::ffi::c_char = c"dltensor".as_ptr() as *const std::ffi::c_char;

    // PyCapsule へ（所有権を Python に渡す）
    //    let mt_box = Box::new(managed);
    //    let raw_ptr = Box::into_raw(mt_box) as *mut std::ffi::c_void;

    //    // Capsule 名は "dltensor"
    //    let capsule = PyCapsule::new(py, raw_ptr, dlpack::DLPACK_CAPSULE_NAME)?;
    //    Ok(capsule)

    unsafe {
        let cap_ptr = pyo3::ffi::PyCapsule_New(
            raw_managed as *mut std::os::raw::c_void,
            // name.as_ptr() as *const std::os::raw::c_char,
            name_ptr,
            Some(capsule_destructor),
        );
        pyo3::Py::<PyAny>::from_owned_ptr(py, cap_ptr)
    }
}

pub fn is_equal<T: ToDataTypeCode>(dt: &DataType) -> bool {
    if T::category() != dt.code {
        return false;
    }
    if dt.bits != std::mem::size_of::<T>() as u8 * 8 {
        return false;
    }
    if dt.lanes != 1 {
        return false;
    }
    true
}

#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err($msg.to_string());
        }
    };
    ($cond:expr $(,)?) => {
        if !$cond {
            return Err(format!("ensure! failed: {}", stringify!($cond)));
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            return Err(format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! ensure_eq {
    ($left:expr, $right:expr $(,)?) => {
        if $left != $right {
            return Err(format!(
                "assertion failed: left = `{:?}`, right = `{:?}`",
                $left, $right
            ));
        }
    };
    ($left:expr, $right:expr, $($arg:tt)*) => {
        if $left != $right {
            return Err(format!(
                "assertion failed: left = `{:?}`, right = `{:?}`: {}",
                $left, $right, format!($($arg)*)
            ));
        }
    };
}

pub fn check_2d_tensor<T: ToDataTypeCode>(
    t: &Tensor,
    d0: i64,
    d1: i64,
    device_type: dlpack::DeviceTypeCode,
) -> Result<(), String> {
    ensure_eq!(t.byte_offset, 0, "{}", t.byte_offset);
    let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
    ensure_eq!(shape.len(), 2, "{}", shape.len());
    ensure_eq!(shape[0], d0);
    ensure_eq!(shape[1], d1);
    ensure!(is_equal::<T>(&t.dtype));
    ensure!(unsafe { is_tensor_c_contiguous(t) });
    ensure_eq!(t.ctx.device_type, device_type);
    Ok(())
}

pub fn check_1d_tensor<T: ToDataTypeCode>(
    t: &Tensor,
    d0: i64,
    device_type: dlpack::DeviceTypeCode,
) -> Result<(), String> {
    ensure_eq!(t.byte_offset, 0);
    let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
    ensure_eq!(shape.len(), 1);
    if d0 != -1 {
        ensure_eq!(shape[0], d0);
    }
    ensure!(is_equal::<T>(&t.dtype), "the data type is different");
    ensure!(unsafe { is_tensor_c_contiguous(t) });
    ensure_eq!(t.ctx.device_type, device_type);
    Ok(())
}

pub fn get_shape_tensor(t: &Tensor, i_dim: usize) -> i64 {
    assert!(i_dim < t.ndim as usize);
    let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
    shape[i_dim]
}
