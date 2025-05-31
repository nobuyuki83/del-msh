import typing
import numpy


class WavefrontObj:

    def __init__(self):
        self.vtxxyz2xyz = None
        self.vtxuv2uv = None
        self.vtxnrm2nrm = None
        self.elem2idx = None
        self.idx2vtxxyz = None
        self.idx2vtxuv = None
        self.idx2vtxnrm = None
        self.elem2group = None
        self.group2name = None
        self.elem2mtl = None
        self.mtl2name = None
        self.mtl_file_name = None

    def centerize(self):
        self.vtxxyz2xyz[:] -= (self.vtxxyz2xyz.max(axis=0) + self.vtxxyz2xyz.min(axis=0)) * 0.5

    def normalize_size(self, scale=1.0):
        self.vtxxyz2xyz *= scale / (self.vtxxyz2xyz.max(axis=0) - self.vtxxyz2xyz.min(axis=0)).max()

    def edge2vtxxyz(self):
        from .del_msh import edge2vtx_polygon_mesh
        return edge2vtx_polygon_mesh(self.elem2idx, self.idx2vtxxyz, self.vtxxyz2xyz.shape[0])

    def tri2vtxxyz(self):
        from .del_msh import triangles_from_polygon_mesh
        return triangles_from_polygon_mesh(self.elem2idx, self.idx2vtxxyz)

    def triangle_mesh_with_uv(self):
        from .del_msh import triangles_from_polygon_mesh, unify_two_indices_of_triangle_mesh

        tri2vtxxyz = triangles_from_polygon_mesh(self.elem2idx, self.idx2vtxxyz)
        tri2vtxuv = triangles_from_polygon_mesh(self.elem2idx, self.idx2vtxuv)

        tri2uni, uni2vtxxyz, uni2vtxuv = unify_two_indices_of_triangle_mesh(tri2vtxxyz, tri2vtxuv)

        uni2xyz = numpy.ndarray((uni2vtxxyz.shape[0], 3), self.vtxxyz2xyz.dtype)
        uni2xyz[:, :] = self.vtxxyz2xyz[uni2vtxxyz[:], :]

        uni2uv = numpy.ndarray((uni2vtxuv.shape[0], 2), self.vtxuv2uv.dtype)
        uni2uv[:, :] = self.vtxuv2uv[uni2vtxuv[:], :]
        return tri2uni, uni2xyz, uni2uv, uni2vtxxyz, uni2vtxuv

    def polygon_mesh_with_normal(self):
        from .del_msh import unify_two_indices_of_polygon_mesh
        idx2uni, uni2vtxxyz, uni2vtxnrm = unify_two_indices_of_polygon_mesh(
            self.elem2idx, self.idx2vtxxyz, self.idx2vtxnrm)

        uni2xyz = numpy.ndarray((uni2vtxxyz.shape[0], 3), self.vtxxyz2xyz.dtype)
        uni2xyz[:, :] = self.vtxxyz2xyz[uni2vtxxyz[:], :]

        uni2nrm = numpy.ndarray((uni2vtxnrm.shape[0], 3), self.vtxnrm2nrm.dtype)
        uni2nrm[:, :] = self.vtxnrm2nrm[uni2vtxnrm[:], :]
        return idx2uni, uni2xyz, uni2nrm, uni2vtxxyz, uni2vtxnrm

    def extract_polygon_mesh_of_material(self, mtl_idx):
        from .del_msh import extract_flagged_polygonal_element
        return extract_flagged_polygonal_element(
            self.elem2idx, self.idx2vtxxyz, self.elem2mtl[:] == mtl_idx)


def load(file_path: str, is_centerize=False, normalized_size: typing.Optional[float] = None):
    from .del_msh import load_wavefront_obj
    o = WavefrontObj()
    o.vtxxyz2xyz, o.vtxuv2uv, o.vtxnrm2nrm, \
        o.elem2idx, o.idx2vtxxyz, o.idx2vtxuv, o.idx2vtxnrm, \
        o.elem2group, o.group2name, \
        o.elem2mtl, o.mtl2name, o.mtl_file_name = load_wavefront_obj(file_path)
    if is_centerize:
        o.centerize()
    if isinstance(normalized_size, float):
        o.normalize_size(normalized_size)
    return o


def read_material(path: str):
    with open(path) as f:
        dict_mtl = {}
        cur_mtl = {}
        cur_name = ""
        for line in f:
            if line.startswith('#'):
                continue
            words = line.split()
            if len(words) == 2 and words[0] == 'newmtl':
                cur_name = words[1]
                cur_mtl = {}
            if len(words) == 0 and cur_name != "":
                dict_mtl[cur_name] = cur_mtl
            if len(words) == 4 and words[0] == 'Kd':
                cur_mtl['Kd'] = (float(words[1]), float(words[2]), float(words[3]))
    return dict_mtl


def save(path: str, elem2vtx, vtx2xyz):
    from .del_msh import save_wavefront_obj_for_uniform_mesh
    if vtx2xyz.dtype != numpy.float32:
        vtx2xyz = vtx2xyz.astype(numpy.float32)
    save_wavefront_obj_for_uniform_mesh(path, elem2vtx, vtx2xyz)
