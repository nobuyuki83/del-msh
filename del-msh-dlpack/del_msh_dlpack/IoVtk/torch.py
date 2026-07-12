import torch


def write_velocity_on_staggered_grid(
        path: str,
        hedge2vy: torch.Tensor,
        vedge2vx: torch.Tensor):
    """Write staggered-grid velocity to a VTK points file.

    Staggered MAC grid convention:
      hedge2vy[j, i]  : y-velocity on horizontal edge between rows j and j+1, column i
                        edge center at (i+0.5, j+1.0)
      vedge2vx[j, i]  : x-velocity on vertical edge between columns i and i+1, row j
                        edge center at (i+1.0, j+0.5)
    """
    img_w = hedge2vy.shape[1]
    img_h = hedge2vy.shape[0] + 1
    assert vedge2vx.shape == (img_h, img_w - 1)

    # --- horizontal edges: center at (i+0.5, j+1.0, 0), velocity (0, vy, 0) ---
    jj_h, ii_h = torch.meshgrid(
        torch.arange(img_h - 1, dtype=torch.float32),
        torch.arange(img_w, dtype=torch.float32),
        indexing='ij')  # shapes (H-1, W)
    h_xyz = torch.stack([
        (ii_h + 0.5).flatten(),
        (jj_h + 1.0).flatten(),
        torch.zeros((img_h - 1) * img_w, dtype=torch.float32),
    ], dim=1)
    h_vel = torch.stack([
        torch.zeros((img_h - 1) * img_w, dtype=torch.float32),
        hedge2vy.float().flatten(),
        torch.zeros((img_h - 1) * img_w, dtype=torch.float32),
    ], dim=1)

    # --- vertical edges: center at (i+1.0, j+0.5, 0), velocity (vx, 0, 0) ---
    jj_v, ii_v = torch.meshgrid(
        torch.arange(img_h, dtype=torch.float32),
        torch.arange(img_w - 1, dtype=torch.float32),
        indexing='ij')  # shapes (H, W-1)
    v_xyz = torch.stack([
        (ii_v + 1.0).flatten(),
        (jj_v + 0.5).flatten(),
        torch.zeros(img_h * (img_w - 1), dtype=torch.float32),
    ], dim=1)
    v_vel = torch.stack([
        vedge2vx.float().flatten(),
        torch.zeros(img_h * (img_w - 1), dtype=torch.float32),
        torch.zeros(img_h * (img_w - 1), dtype=torch.float32),
    ], dim=1)

    vtx2xyz = torch.cat([h_xyz, v_xyz], dim=0).contiguous()
    vtx2velocity = torch.cat([h_vel, v_vel], dim=0).contiguous()

    write_points_with_velocity(str(path), vtx2xyz, vtx2velocity)


def write_points_with_velocity(path: str, vtx2xyz: torch.Tensor, vtx2velocity: torch.Tensor):
    from .. import util_torch
    from ..del_msh_dlpack import io_vtk_write_points_with_velocity
    assert vtx2xyz.ndim == 2 and vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == torch.float32
    assert vtx2velocity.shape == vtx2xyz.shape
    assert vtx2velocity.dtype == torch.float32
    vtx2xyz = vtx2xyz.contiguous()
    vtx2velocity = vtx2velocity.contiguous()
    io_vtk_write_points_with_velocity(
        path,
        util_torch.to_dlpack_safe(vtx2xyz, 0),
        util_torch.to_dlpack_safe(vtx2velocity, 0),
    )


def write_mix_mesh(
        path_file: str,
        vtx2xyz: torch.Tensor,
        tet2vtx: torch.Tensor,
        pyrmd2vtx: torch.Tensor,
        prism2vtx: torch.Tensor,
        hex2vtx: torch.Tensor):
    """Save a mixed-element mesh to a VTK file.

    Args:
        path_file: output file path
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
        tet2vtx: (num_tet, 4) uint32 - tetrahedron connectivity
        pyrmd2vtx: (num_pyrmd, 5) uint32 - pyramid connectivity
        prism2vtx: (num_prism, 6) uint32 - prism connectivity
    """
    #
    from .. import util_torch
    num_vtx = vtx2xyz.shape[0]
    num_tet = tet2vtx.shape[0]
    num_pyrmd = pyrmd2vtx.shape[0]
    num_prism = prism2vtx.shape[0]
    num_hex = hex2vtx.shape[0]
    #
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(tet2vtx, (num_tet, 4), torch.uint32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(pyrmd2vtx, (num_pyrmd, 5), torch.uint32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(prism2vtx, (num_prism, 6), torch.uint32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(hex2vtx, (num_hex, 8), torch.uint32, torch.device("cpu"))
    #
    from .. import IoVtk
    IoVtk.write_mix_mesh(
        path_file,
        util_torch.to_dlpack_safe(vtx2xyz, 0),
        util_torch.to_dlpack_safe(tet2vtx, 0),
        util_torch.to_dlpack_safe(pyrmd2vtx, 0),
        util_torch.to_dlpack_safe(prism2vtx, 0),
        util_torch.to_dlpack_safe(hex2vtx, 0)
    )
