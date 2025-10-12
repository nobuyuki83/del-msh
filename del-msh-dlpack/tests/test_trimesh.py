import math
import os.path

#
import numpy
import torch

#
import del_msh_numpy.TriMesh
import del_msh_numpy.BVH
import del_msh_dlpack.Raycast
import del_msh_dlpack.TriMesh3.np
import del_msh_dlpack.TriMesh3.pt


def test_01():
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.sphere(0.8, 128, 64)
    tri2vtx = tri2vtx.astype(numpy.uint32)
    np_tri2normal = del_msh_dlpack.TriMesh3.np.tri2normal(tri2vtx, vtx2xyz)
    tri2area = numpy.linalg.norm(np_tri2normal, axis=1)
    area = tri2area.sum() * 0.5
    area0 = 0.8 * 0.8 * 4.0 * numpy.pi
    assert abs(area - area0) / area0 < 0.001
    rng = numpy.random.default_rng(seed=42)
    np_dw_tri2nrm = rng.random(tri2vtx.shape).astype(numpy.float32)
    np_dw_vtx2xyz = del_msh_dlpack.TriMesh3.np.bwd_tri2normal(
        tri2vtx, vtx2xyz, np_dw_tri2nrm
    )
    #
    tri2vtx = torch.from_numpy(tri2vtx)
    vtx2xyz = torch.from_numpy(vtx2xyz)
    ptcpu_tri2normal = del_msh_dlpack.TriMesh3.pt.tri2normal(tri2vtx, vtx2xyz)
    ptcpu_area = ptcpu_tri2normal.norm(dim=1).sum() * 0.5
    assert abs(area - ptcpu_area.item()) < 1.0e-20
    ptcpu_dw_vtx2xyz = del_msh_dlpack.TriMesh3.pt.bwd_tri2normal(
        tri2vtx, vtx2xyz, torch.from_numpy(np_dw_tri2nrm)
    )
    assert numpy.linalg.norm(np_dw_vtx2xyz - ptcpu_dw_vtx2xyz.numpy()) < 1.0e-20
    if torch.cuda.is_available():
        print('test "tri2nrm" and "dw_tri2nrm" on gpu')
        ptcuda_tri2nrm = del_msh_dlpack.TriMesh3.pt.tri2normal(
            tri2vtx.cuda(), vtx2xyz.cuda()
        )
        # print(ptcuda_tri2normal)
        n0 = torch.norm(ptcpu_tri2normal - ptcuda_tri2nrm.cpu())
        n1 = torch.norm(ptcuda_tri2nrm)
        assert n0 / n1 < 1.0e-7, n0 / n1
        ptcuda_dw_vtx2xyz = del_msh_dlpack.TriMesh3.pt.bwd_tri2normal(
            tri2vtx.cuda(), vtx2xyz.cuda(), torch.from_numpy(np_dw_tri2nrm).cuda()
        )
        n0 = torch.norm(ptcuda_dw_vtx2xyz.cpu() - ptcpu_dw_vtx2xyz)
        n1 = torch.norm(ptcpu_dw_vtx2xyz)
        assert n0 / n1 < 2.0e-7, n0 / n1


def test_02():
    """
    test raycast
    """
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.sphere(0.8, 64, 32)
    bvhnodes = del_msh_numpy.TriMesh.bvhnodes_tri(tri2vtx, vtx2xyz)
    assert bvhnodes.shape[1] == 3
    bvhnode2aabb = del_msh_numpy.BVH.aabb_uniform_mesh(
        tri2vtx, vtx2xyz, bvhnodes, aabbs=None, vtx2xyz1=None
    )
    assert bvhnode2aabb.shape[1] == 6
    transform_ndc2world = numpy.eye(4, dtype=numpy.float32)
    #
    pix2tri = numpy.ndarray(shape=(300, 300), dtype=numpy.uint64)

    del_msh_dlpack.Raycast.pix2tri(
        pix2tri, tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world
    )

    mask = pix2tri != numpy.iinfo(numpy.uint64).max
    num_true = numpy.count_nonzero(mask)
    ratio0 = float(num_true) / float(mask.shape[0] * mask.shape[1])
    ratio1 = 0.8 * 0.8 * numpy.pi * 0.25
    assert abs(ratio0 - ratio1) < 0.00013

    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent
    # print(path.resolve() )

    from PIL import Image

    img = Image.fromarray((mask.astype(numpy.uint8) * 255))
    file_path = path / "target/del_msh_delpack__pix2tri.png"
    img.save(file_path)
    #

    """
    pix2depth = numpy.zeros(shape=(300, 300), dtype=numpy.float32)
    Raycast.pix2depth(
        pix2depth,
        tri2vtx=tri2vtx,
        vtx2xyz=vtx2xyz,
        bvhnodes=bvhnodes,
        bvhnode2aabb=bvhnode2aabb,
        transform_ndc2world = transform_ndc2world)
    img = Image.fromarray((pix2depth * 255.).astype(numpy.uint8))
    img.save("../target/del_msh_numpy__pix2depth.png")
    """


def test_03():
    tri2vtx, vtx2xyz0 = del_msh_numpy.TriMesh.sphere(0.8, 4, 4)
    tri2vtx = tri2vtx.astype(numpy.uint32)
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz0.shape[0]
    rng = numpy.random.default_rng(seed=42)
    dw_tri2nrm = rng.random((num_tri, 3)).astype(numpy.float32)
    tri2nrm0 = del_msh_dlpack.TriMesh3.np.tri2normal(tri2vtx, vtx2xyz0)
    loss0 = numpy.tensordot(dw_tri2nrm, tri2nrm0)
    dw_vtx2xyz = del_msh_dlpack.TriMesh3.np.bwd_tri2normal(
        tri2vtx, vtx2xyz0, dw_tri2nrm
    )
    eps = 1.0e-3
    for i_vtx in range(0, num_vtx):
        for i_dim in range(0, 3):
            vtx2xyz1 = vtx2xyz0.copy()
            vtx2xyz1[i_vtx, i_dim] += eps
            tri2nrm1 = del_msh_dlpack.TriMesh3.np.tri2normal(tri2vtx, vtx2xyz1)
            loss1 = numpy.tensordot(dw_tri2nrm, tri2nrm1)
            diff_num = (loss1 - loss0) / eps
            diff_ana = dw_vtx2xyz[i_vtx, i_dim]
            ratio = abs(diff_num - diff_ana) / (abs(diff_ana) + 0.001)
            assert ratio < 0.007


def test_04():
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.sphere(0.8, 128, 64)
    tri2vtx, vtx2xyz = (
        torch.from_numpy(tri2vtx).to(torch.uint32),
        torch.from_numpy(vtx2xyz),
    )
    dw_tri2nrm = torch.rand(size=tri2vtx.shape, dtype=torch.float32)
    dw_vtx2xyz0 = del_msh_dlpack.TriMesh3.pt.bwd_tri2normal(
        tri2vtx, vtx2xyz, dw_tri2nrm
    )
    #
    vtx2xyz = torch.nn.Parameter(vtx2xyz)
    optimizer = torch.optim.Adam([vtx2xyz], lr=0.001)
    optimizer.zero_grad()
    tri2nrm = del_msh_dlpack.TriMesh3.pt.Tri2Normal.apply(tri2vtx, vtx2xyz)
    loss0 = torch.tensordot(tri2nrm, dw_tri2nrm)
    loss0.backward()
    print(loss0)
    dw_vtx2xyz1 = vtx2xyz.grad
    n0 = torch.norm(dw_vtx2xyz0 - dw_vtx2xyz1)
    assert n0 == 0.0, n0
    #
    if torch.cuda.is_available():
        tri2vtx, vtx2xyz = tri2vtx.cuda(), vtx2xyz.cuda()
        dw_tri2nrm = dw_tri2nrm.cuda()
        vtx2xyz = torch.nn.Parameter(vtx2xyz)
        optimizer = torch.optim.Adam([vtx2xyz], lr=0.001)
        optimizer.zero_grad()
        tri2nrm = del_msh_dlpack.TriMesh3.pt.Tri2Normal.apply(tri2vtx.cuda(), vtx2xyz)
        loss1 = torch.tensordot(tri2nrm, dw_tri2nrm)
        loss1.backward()
        dw_vtx2xyz2 = vtx2xyz.grad
        n0 = torch.norm(dw_vtx2xyz0 - dw_vtx2xyz2.cpu())
        n1 = torch.norm(dw_vtx2xyz0)
        assert n0 / n1 < 1.0e-6, n0 / n1
        print(loss1)
        print(dw_vtx2xyz2)
        print(dw_vtx2xyz1)
