import torch
from typing import Tuple

import del_msh_dlpack.Pix2Tri.torch as Pix2Tri


# Function to generate axis-aligned lighting directions and color permutations
def generate_lighting(
    low: float, mid: float, high: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate axis-aligned lighting directions and corresponding RGB colors using permutations of intensity levels.

    Args:
        low: Lower bound intensity.
        mid: Medium intensity.
        high: High intensity.

    Returns:
        A tuple of:
            - L_dirs: Tensor of shape (6, 3) representing lighting directions.
            - L_colors: Tensor of shape (6, 3) representing corresponding RGB colors.
    """
    light2dir = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Positive X
            [0.0, 1.0, 0.0],  # Positive Y
            [0.0, 0.0, 1.0],  # Positive Z
            [-1.0, 0.0, 0.0],  # Negative X
            [0.0, -1.0, 0.0],  # Negative Y
            [0.0, 0.0, -1.0],  # Negative Z
        ],
    )

    light2rgb = torch.tensor(
        [
            [high, mid, low],
            [low, high, mid],
            [mid, low, high],
            [low, mid, high],
            [high, low, mid],
            [mid, high, low],
        ],
    )

    return light2dir, light2rgb


def render_lambertian_shading_gouraud(
    tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world, L_dirs, L_colors, pix2tri
):
    device = tri2vtx.device
    #
    assert len(L_dirs) == len(L_colors)
    background_pix = pix2tri == torch.iinfo(torch.uint32).max
    pix2rgb_bg = torch.zeros(
        (pix2tri.shape[0], pix2tri.shape[1], 3), dtype=torch.float32, device=device
    )

    vtx2diffuse = torch.zeros_like(vtx2xyz)
    for i in range(len(L_dirs)):
        dir = torch.nn.functional.normalize(L_dirs[i], dim=0)
        ndotl = torch.clamp((vtx2nrm @ dir).unsqueeze(-1), min=0.0)
        vtx2diffuse += ndotl * L_colors[i]

    pix2diffuse = Pix2Tri.interpolate(
        pix2tri, tri2vtx, vtx2xyz, vtx2diffuse, transform_ndc2world
    )
    pix2rgb = pix2diffuse
    pix2rgb = torch.where(background_pix.unsqueeze(-1), pix2rgb_bg, pix2rgb)
    return pix2rgb


def render_lambertian_shading_phong(
    tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world, lighting, base_color, pix2tri
):
    device = tri2vtx.device
    background_pix = pix2tri == torch.iinfo(torch.uint32).max
    pix2rgb_bg = torch.zeros(
        (pix2tri.shape[0], pix2tri.shape[1], 3), dtype=torch.float32, device=device
    )
    pix2nrm = Pix2Tri.interpolate(
        pix2tri, tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world
    )
    pix2nrm_bg = pix2nrm.new_tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    pix2nrm = torch.where(
        background_pix.unsqueeze(-1),
        pix2nrm_bg,
        pix2nrm,
    )
    """
    return (pix2nrm+1.0)*0.5
    pix2nrm = torch.nn.functional.normalize(
        pix2nrm,
        dim=-1,
    )
    """

    pix2rgb = torch.where(
        background_pix.unsqueeze(-1),
        pix2rgb_bg,
        (pix2nrm + 1.0) * 0.5,
    )
    return pix2rgb
    #
    # img = (((pix2nrm + 1.) * 0.5).detach().numpy() * 255).clip(0, 255).astype('uint8')
    # path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__pix2tri1.png"
    # path0.parent.mkdir(exist_ok=True)
    # Image.fromarray(img).save(path0)
    #
    light_dir = torch.tensor(light_dir).view(1, 1, 3)
    base_color = torch.tensor(base_color).view(1, 1, 3)
    pix2ndotl = torch.clamp(
        torch.sum(pix2nrm * light_dir, dim=-1, keepdim=True),
        min=0.0,
    )
    pix2diffuse = pix2ndotl * base_color
    pix2rgb = pix2diffuse
    pix2rgb = torch.where(background_pix.unsqueeze(-1), pix2rgb_bg, pix2rgb)
    return pix2rgb
