pub const AABB3_FROM_VTX2XYZ: &str =
    include_str!(concat!(env!("OUT_DIR"), "/aabb3_from_vtx2xyz.ptx"));
pub const BVHNODE2AABB: &str = include_str!(concat!(env!("OUT_DIR"), "/bvhnode2aabb.ptx"));
pub const BVHNODES_MORTON: &str = include_str!(concat!(env!("OUT_DIR"), "/bvhnodes_morton.ptx"));
pub const EDGE2VTX: &str = include_str!(concat!(env!("OUT_DIR"), "/edge2vtx.ptx"));
pub const PIX2TRI: &str = include_str!(concat!(env!("OUT_DIR"), "/pix2tri.ptx"));
