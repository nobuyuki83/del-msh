import numpy as np

from del_msh_numpy import TriMesh, AlembicExporter


def test_02():
    tri2vtx, vtx2xyz = TriMesh.torus()
    export = AlembicExporter.MeshVtxAnimation(tri2vtx, vtx2xyz)
    for t in range(1, 300):
        vtx2xyz1 = vtx2xyz.copy() * np.cos( float(t) * 0.1 )
        export.add_frame(vtx2xyz1)
    export.write("../target/hoge.abc")
