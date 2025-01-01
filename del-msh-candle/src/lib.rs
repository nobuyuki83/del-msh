macro_rules! get_cpu_slice_from_tensor {
    ($slice: ident, $storage: ident, $tensor: expr, $t: ty) => {
        let $storage = $tensor.storage_and_layout().0;
        let $slice = match $storage.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<$t>()?,
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_from_tensor {
    ($slc: ident, $storage: ident, $layout: ident, $tnsr: expr, $t: ty) => {
        let ($storage, $layout) = $tnsr.storage_and_layout();
        let $slc = match $storage.deref() {
            candle_core::Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<$t>()?,
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_from_storage_u32 {
    ($slice: ident, $device: ident, $storage: expr) => {
        let CudaStorage { slice, device } = $storage;
        let ($slice, $device) = match slice {
            CudaStorageSlice::U32(slice) => (slice, device),
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_from_storage_f32 {
    ($slice: ident, $device: ident, $storage: expr) => {
        let CudaStorage { slice, device } = $storage;
        let ($slice, $device) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
    };
}

// -----------------------------------------------

pub mod perturb_tensor;
//
pub mod bvhnode2aabb;
pub mod bvhnodes_morton;
pub mod elem2center;
//
pub mod diffcoord_polyloop2;
pub mod diffcoord_trimesh3;
pub mod edge2vtx_trimesh3;

pub mod polygonmesh2_to_cogs;
pub mod voronoi2;
pub mod vtx2xyz_to_edgevector;
