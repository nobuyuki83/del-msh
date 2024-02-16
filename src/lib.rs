// topology
pub mod vtx2elem;
pub mod vtx2vtx;
pub mod elem2elem;
pub mod edge2vtx;
pub mod tri2vtx;
pub mod unify_index;

// functions specific to type of mesh
pub mod map_idx;
pub mod polyloop;
pub mod polyloop2;
pub mod polyloop3;
pub mod polyline;
pub mod polyline3;
pub mod trimesh2;
pub mod trimesh3;
pub mod trimesh3_search_bruteforce;
pub mod trimesh3_primitive;
pub mod quadmesh;
pub mod vtx2xyz;

// misc functions general to mesh type
pub mod elem2group;
pub mod elem2center;
pub mod sampling;
pub mod transform;
pub mod extract;
pub mod unindex;
pub mod dijkstra;

// io
pub mod io_obj;
pub mod io_off;
pub mod io_vtk;
pub mod io_nas;

// search
pub mod kdtree2;
pub mod bvh3;
pub mod bvh3_topology_morton;
pub mod bvh;
pub mod bvh3_topology_topdown;

// self intersection
pub mod trimesh3_intersection;
pub mod trimesh3_intersection_time;
pub mod trimesh3_proximity;
pub mod trimesh3_move_avoid_intersection;