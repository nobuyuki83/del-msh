// use cudarc::driver::{CudaDevice, CudaSlice};

#[cfg(feature = "cuda")]
pub mod elem2center;

#[cfg(feature = "cuda")]
pub mod vtx2xyz;

#[cfg(feature = "cuda")]
pub mod bvhnodes_morton;

#[cfg(feature = "cuda")]
pub mod bvhnode2aabb;

#[cfg(feature = "cuda")]
pub mod edge2vtx_contour;

#[cfg(feature = "cuda")]
pub mod pix2depth;

#[cfg(feature = "cuda")]
pub mod pix2tri;

#[cfg(feature = "cuda")]
pub mod silhouette;
