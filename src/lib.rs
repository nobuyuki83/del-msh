// topology
pub mod vtx2elem;
pub mod vtx2vtx;
pub mod elem2elem;
pub mod line2vtx;
pub mod tri2vtx;
pub mod unify_index;

// mesh from here
pub mod edgeloop;
pub mod polyline;
pub mod trimesh;
pub mod group;
pub mod sampling;
pub mod transform;
pub mod extract;

// io
pub mod io_obj;
pub mod io_off;

pub mod primitive
;
pub mod unindex;
pub mod dijkstra;