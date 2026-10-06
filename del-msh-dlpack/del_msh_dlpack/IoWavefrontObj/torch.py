import torch


def save_trimesh3(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor, path_file: str):
    """Save a triangle mesh to a Wavefront OBJ file.

    Args:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity (CPU only)
        vtx2xyz: (num_vtx, 3) float32 - vertex positions (CPU only)
        path_file: output file path
    """
    assert tri2vtx.device.type == "cpu"
    assert vtx2xyz.device.type == "cpu"

    from .. import IoWavefrontObj

    IoWavefrontObj.save_trimesh3(
        tri2vtx.__dlpack__(), vtx2xyz.detach().__dlpack__(), path_file
    )


def save_points(xyz: torch.Tensor, path: str):
    """
    Save a point cloud as a Wavefront OBJ file (vertices only, no faces).

    Args:
        xyz: (N, 3) float tensor of 3D point positions.
        path: Output file path.
    """
    xyz = xyz.detach().cpu().to(torch.float32).numpy()
    with open(path, "w", encoding="utf-8") as f:
        f.write("# OBJ point cloud\n")
        for x, y, z in xyz:
            f.write(f"v {x} {y} {z}\n")  # 頂点定義のみ
    print(f"Saved {len(xyz)} points to {path}")
