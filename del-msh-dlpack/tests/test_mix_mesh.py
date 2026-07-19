import pathlib
import pytest
import torch
import del_msh_dlpack.MixMesh3.torch as MixMesh3
import del_msh_dlpack.PolyhedronMesh.torch as PolyhedronMesh
import del_msh_dlpack.IoVtk.torch as IoVtk


def hoge():
    path = pathlib.Path(__file__).parent.parent.parent / "asset" / "cfd_mesh.txt"
    vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx, hex2vtx, vtx2velo, vtx2press = (
        MixMesh3.load_cfd_mesh(str(path))
    )
    path_vtk = pathlib.Path(__file__).parent.parent.parent / "target" / "mix_mesh.vtk"
    IoVtk.write_mix_mesh(
        str(path_vtk),
        vtx2xyz,
        tet2vtx.to(torch.uint32),
        pyrmd2vtx.to(torch.uint32),
        prism2vtx.to(torch.uint32),
        hex2vtx.to(torch.uint32),
    )
    elem2idx_offset, idx2vtx = MixMesh3.to_polyhedral_mesh(
        tet2vtx.to(torch.uint32),
        pyrmd2vtx.to(torch.uint32),
        prism2vtx.to(torch.uint32),
        hex2vtx.to(torch.uint32),
    )
    elem2volume = PolyhedronMesh.make_elem2volume(elem2idx_offset, idx2vtx, vtx2xyz)

    # elem2idx_offset1 = elem2idx_offset
    # idx2vtx1 = idx2vtx
    # vtx2xyz1 = vtx2xyz

    elem2idx_offset1, idx2vtx1, vtx2xyz1 = PolyhedronMesh.subdivide(
        elem2idx_offset, idx2vtx, vtx2xyz
    )

    elem2idx_offset1, idx2vtx1, vtx2xyz1 = PolyhedronMesh.subdivide(
        elem2idx_offset1, idx2vtx1, vtx2xyz1
    )
    elem2idx_offset1, idx2vtx1, vtx2xyz1 = PolyhedronMesh.subdivide(
        elem2idx_offset1, idx2vtx1, vtx2xyz1
    )
    elem2volume1 = PolyhedronMesh.make_elem2volume(elem2idx_offset1, idx2vtx1, vtx2xyz1)
    print("elem2volume", elem2volume1.shape)
    assert abs(elem2volume1.sum() - elem2volume.sum()) < 1e-6
    #
    # per-vertex values: linear transform of position, v = A @ xyz + b
    A = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 1.0, -1.0], [2.0, -1.0, 0.0]], dtype=torch.float32
    )
    b = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    vtx2value1 = (vtx2xyz1 @ A.T) + b  # (num_vtx, 3)
    #
    num_query = 200
    lo = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32)
    hi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    torch.manual_seed(0)
    wtx2xyz = torch.rand(num_query, 3, dtype=torch.float32) * (hi - lo) + lo
    wtx2value1_expected = (wtx2xyz @ A.T) + b
    return (
        elem2idx_offset1,
        idx2vtx1,
        vtx2xyz1,
        vtx2value1,
        wtx2xyz,
        wtx2value1_expected,
    )


def test_01():
    elem2idx_offset, idx2vtx, vtx2xyz, vtx2value, wtx2xyz, wtx2value1_expected = hoge()
    bvhnodes, bvhnode2aabb = PolyhedronMesh.make_bvhnodes_bvhnode2aabb(
        elem2idx_offset, idx2vtx, vtx2xyz
    )
    print("bvhnodes:", bvhnodes.shape)
    print("bvhnode2aabb:", bvhnode2aabb.shape)
    wtx2elem, wtx2param = PolyhedronMesh.search_elem_contain_points(
        elem2idx_offset, idx2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, wtx2xyz
    )
    print("wtx2elem:", wtx2elem.shape)
    print("wtx2param:", wtx2param.shape)
    wtx2value = PolyhedronMesh.interpolate_values_at_points(
        elem2idx_offset, idx2vtx, vtx2value, wtx2elem, wtx2param
    )
    # For points inside the mesh the interpolated value should match the linear transform exactly
    inside = wtx2elem != torch.iinfo(torch.uint32).max
    err = (wtx2value[inside] - wtx2value1_expected[inside]).abs()
    assert err.max().item() < 1.0e-4
    if torch.cuda.is_available():
        d_elem2idx_offset = elem2idx_offset.cuda()
        d_idx2vtx = idx2vtx.cuda()
        d_vtx2xyz = vtx2xyz.cuda()
        d_wtx2xyz = wtx2xyz.cuda()
        d_vtx2value = vtx2value.cuda()
        d_bvhnodes, d_bvhnode2aabb = PolyhedronMesh.make_bvhnodes_bvhnode2aabb(
            d_elem2idx_offset, d_idx2vtx, d_vtx2xyz
        )
        assert torch.equal(d_bvhnodes.cpu(), bvhnodes)
        d_wtx2elem, d_wtx2param = PolyhedronMesh.search_elem_contain_points(
            d_elem2idx_offset,
            d_idx2vtx,
            d_vtx2xyz,
            d_bvhnodes,
            d_bvhnode2aabb,
            d_wtx2xyz,
        )
        assert torch.equal(d_wtx2elem.cpu(), wtx2elem)
        assert (d_wtx2param.cpu() - wtx2param).abs().max() < 1.0e-7
        d_wtx2value = PolyhedronMesh.interpolate_values_at_points(
            d_elem2idx_offset, d_idx2vtx, d_vtx2value, d_wtx2elem, d_wtx2param
        )
        print((d_wtx2value.cpu() - wtx2value).max() < 1.0e-7)


def test_file_no_exist():
    path = pathlib.Path(__file__).parent.parent.parent / "asset" / "cfd_mesh.txt"
    vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx, hex2vtx, vtx2velo, vtx2press = (
        MixMesh3.load_cfd_mesh(str(path))
    )
    path_vtk = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "no-exist"
        / "mix_mesh.vtk"
    )
    with pytest.raises(OSError) as exc_info:
        IoVtk.write_mix_mesh(
            str(path_vtk),
            vtx2xyz,
            tet2vtx.to(torch.uint32),
            pyrmd2vtx.to(torch.uint32),
            prism2vtx.to(torch.uint32),
            hex2vtx.to(torch.uint32),
        )
    print("expected error:", exc_info.value)
