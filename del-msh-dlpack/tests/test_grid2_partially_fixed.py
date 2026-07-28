import pathlib

import torch
from PIL import Image

import del_msh_dlpack.Grid2PartiallyFixed.torch as Grid2
from del_msh_numpy.Raycast import pix2tri, pix2depth


def test_nearest_to_fixed_cell():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = 511, 512

    # Random seed pixels with random RGB colours
    torch.manual_seed(0)
    pix2isfix = torch.zeros((img_h, img_w), dtype=torch.uint8)
    pix2rgb_ini = torch.zeros((img_h, img_w, 3), dtype=torch.float32)
    n_fix = 100
    indices = torch.randint(0, img_h * img_w, (n_fix,))
    for idx in indices:
        ih, iw = idx.item() // img_w, idx.item() % img_w
        pix2rgb_ini[ih, iw] = torch.rand(3)
        pix2isfix[ih, iw] = 1

    # --- Gauss-Seidel smoothing (CPU) ---
    pix2rgb_gs = pix2rgb_ini.clone()
    for i_gs in range(8):
        img = (pix2rgb_gs.numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img).save(path_dir / f"grid2_partially_fixed_gs{i_gs}_cpu.png")
        Grid2.smooth_gauss_seidel(pix2isfix, pix2rgb_gs, 100)

    # --- nearest_to_fixed_cell + smoothing with radius (CPU) ---
    pix2nearest, pix2distance = Grid2.nearest_to_fixed_cell(pix2isfix)

    # Paint each pixel with its nearest seed's colour
    pix2rgb_jump = pix2rgb_ini.view(-1, 3)[pix2nearest.view(-1).long()].view(
        img_h, img_w, 3
    )
    pix2rgb_jump = pix2rgb_jump.clone()

    n_iter = 8
    for i_iter in range(n_iter):
        img = (pix2rgb_jump.numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img).save(
            path_dir / f"grid2_partially_fixed_jump{i_iter}_cpu.png"
        )
        ratio = 1.0 - (i_iter + 1) / n_iter
        Grid2.smooth_gauss_seidel_with_radius(
            pix2isfix, pix2distance, ratio, pix2rgb_jump
        )

    if torch.cuda.is_available():
        d_pix2isfix = pix2isfix.cuda()
        d_pix2rgb_gs = pix2rgb_ini.clone().cuda()
        for i_gs in range(8):
            img = (d_pix2rgb_gs.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(img).save(
                path_dir / f"grid2_partially_fixed_gs{i_gs}_gpu.png"
            )
            Grid2.smooth_gauss_seidel(d_pix2isfix, d_pix2rgb_gs, 100)

        d_pix2nearest, d_pix2distance = Grid2.nearest_to_fixed_cell(d_pix2isfix)

        # Paint each pixel with its nearest seed's colour
        d_pix2rgb_jump = (
            pix2rgb_ini.cuda()
            .view(-1, 3)[d_pix2nearest.view(-1).long()]
            .view(img_h, img_w, 3)
        )
        d_pix2rgb_jump = d_pix2rgb_jump.clone()

        for i_iter in range(n_iter):
            img = (d_pix2rgb_jump.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(img).save(
                path_dir / f"grid2_partially_fixed_jump{i_iter}_gpu.png"
            )
            ratio = 1.0 - (i_iter + 1) / n_iter
            Grid2.smooth_gauss_seidel_with_radius(
                d_pix2isfix, d_pix2distance, ratio, d_pix2rgb_jump
            )
