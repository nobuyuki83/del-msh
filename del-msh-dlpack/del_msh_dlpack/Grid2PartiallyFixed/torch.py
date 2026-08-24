import torch

from .. import util_torch


def nearest_to_fixed_cell(
    cell2isfixed: torch.Tensor,
    is_initial: bool = True,
    cell2nearest: torch.Tensor | None = None,
    cell2distance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-pass chamfer distance transform on a 2-D grid.

    Args:
        cell2isfixed: (h, w) uint8; 1 = fixed seed cell, 0 = free.
        is_initial:   if True, initialize outputs from cell2isfixed before sweeping.
        cell2nearest: (h, w) uint32 tensor to fill in-place (allocated if None).
        cell2distance: (h, w) float32 tensor to fill in-place (allocated if None).

    Returns:
        cell2nearest:  (h, w) uint32 - flat row-major index of the nearest seed cell.
        cell2distance: (h, w) float32 - Euclidean distance to that seed.
    """
    device = cell2isfixed.device
    img_h, img_w = cell2isfixed.shape
    #
    if cell2nearest is None:
        cell2nearest = torch.empty((img_h, img_w), dtype=torch.uint32, device=device)
    if cell2distance is None:
        cell2distance = torch.empty((img_h, img_w), dtype=torch.float32, device=device)
    #
    util_torch.assert_shape_dtype_device(
        cell2isfixed, (img_h, img_w), torch.uint8, device
    )
    util_torch.assert_shape_dtype_device(
        cell2nearest, (img_h, img_w), torch.uint32, device
    )
    util_torch.assert_shape_dtype_device(
        cell2distance, (img_h, img_w), torch.float32, device
    )
    #
    cell2isfixed = cell2isfixed.contiguous()
    cell2nearest = cell2nearest.contiguous()
    cell2distance = cell2distance.contiguous()
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream

    from .. import Grid2PartiallyFixed

    Grid2PartiallyFixed.grid2_nearest_to_fixed_cell(
        util_torch.to_dlpack_safe(cell2isfixed, stream_ptr),
        util_torch.to_dlpack_safe(cell2nearest, stream_ptr),
        util_torch.to_dlpack_safe(cell2distance, stream_ptr),
        is_initial,
        stream_ptr,
    )
    return cell2nearest, cell2distance


def smooth_gauss_seidel_naive(
    cell2isfix: torch.Tensor, cell2val: torch.Tensor, num_iter: int
) -> None:
    """One Gauss-Seidel sweep over a 2-D grid, averaging free cells from their 4 neighbours.

    On CPU: sequential sweep (matches del_msh_cpu::smooth_gauss_seidel).
    On CUDA: equivalent red-black sweep (two kernel passes, race-free in-place).

    Args:
        cell2isfix: (h, w) uint8; 1 = fixed (skipped), 0 = free.
        cell2val:   (h, w, num_vdim) float32, modified in-place.
    """
    assert cell2isfix.ndim == 2 and cell2isfix.dtype == torch.uint8
    assert cell2val.ndim == 3 and cell2val.dtype == torch.float32
    assert cell2isfix.shape == cell2val.shape[:2]

    device = cell2isfix.device
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream

    cell2isfix = cell2isfix.contiguous()
    cell2val = cell2val.contiguous()

    from .. import Grid2PartiallyFixed

    Grid2PartiallyFixed.grid2_smooth_gauss_seidel(
        util_torch.to_dlpack_safe(cell2isfix, stream_ptr),
        util_torch.to_dlpack_safe(cell2val, stream_ptr),
        num_iter,
        stream_ptr,
    )


def smooth_gauss_seidel_fast(
    cell2isfixed: torch.Tensor, cell2val: torch.Tensor, n_iter=8
):
    img_w = cell2isfixed.shape[1]
    img_h = cell2isfixed.shape[0]
    num_vdim = cell2val.shape[2]
    device = cell2isfixed.device
    #
    util_torch.assert_shape_dtype_device(
        cell2isfixed, (img_h, img_w), torch.uint8, device
    )
    util_torch.assert_shape_dtype_device(
        cell2val, (img_h, img_w, num_vdim), torch.float32, device
    )

    cell2nearest, cell2distance = nearest_to_fixed_cell(cell2isfixed)

    # Paint each pixel with its nearest seed's colour
    cell2val_nearest = cell2val.view(-1, num_vdim)[cell2nearest.view(-1).long()].view(
        img_h, img_w, num_vdim
    )
    cell2val[:] = cell2val_nearest[:]
    for i_iter in range(n_iter):
        ratio = 1.0 - (i_iter + 1) / n_iter
        smooth_gauss_seidel_with_radius(cell2isfixed, cell2distance, ratio, cell2val)


def smooth_gauss_seidel_with_radius(
    cell2isfixed: torch.Tensor,
    cell2dist: torch.Tensor,
    ratio: float,
    cell2val: torch.Tensor,
) -> None:
    """Jacobi step with a per-cell neighbourhood radius.

    The radius r = floor(cell2dist[i] * ratio).  On CPU this is a sequential
    Gauss-Seidel sweep; on CUDA it is a parallel Jacobi step (one temp-buffer
    ping-pong per call), which converges to the same result.

    Args:
        cell2isfixed: (h, w) uint8; 1 = fixed (skipped), 0 = free.
        cell2dist:    (h, w) float32 - distance to nearest seed.
        ratio:        neighbourhood radius = floor(dist * ratio).
        cell2val:     (h, w, num_vdim) float32, modified in-place.
    """
    assert cell2isfixed.ndim == 2 and cell2isfixed.dtype == torch.uint8
    assert cell2dist.ndim == 2 and cell2dist.dtype == torch.float32
    assert cell2val.ndim == 3 and cell2val.dtype == torch.float32
    assert cell2isfixed.shape == cell2dist.shape == cell2val.shape[:2]

    device = cell2isfixed.device
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream

    cell2isfixed = cell2isfixed.contiguous()
    cell2dist = cell2dist.contiguous()
    cell2val = cell2val.contiguous()

    from .. import Grid2PartiallyFixed

    Grid2PartiallyFixed.grid2_smooth_gauss_seidel_with_radius(
        util_torch.to_dlpack_safe(cell2isfixed, stream_ptr),
        util_torch.to_dlpack_safe(cell2dist, stream_ptr),
        ratio,
        util_torch.to_dlpack_safe(cell2val, stream_ptr),
        stream_ptr,
    )
