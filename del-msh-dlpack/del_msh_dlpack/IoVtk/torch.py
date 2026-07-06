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
        torch.arange(img_w,     dtype=torch.float32),
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
        torch.arange(img_h,     dtype=torch.float32),
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

    vtx2xyz      = torch.cat([h_xyz, v_xyz], dim=0).contiguous()
    vtx2velocity = torch.cat([h_vel, v_vel], dim=0).contiguous()

    from .. import Vtx2Xyz
    Vtx2Xyz.torch.write_vtk_points_with_velocity(str(path), vtx2xyz, vtx2velocity)

