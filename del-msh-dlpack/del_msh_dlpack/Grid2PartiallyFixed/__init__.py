def grid2_nearest_to_fixed_cell(
    cell2isfixed, cell2nearest, cell2distance, is_initial: bool, stream_ptr: int = 0
):
    from ..del_msh_dlpack import grid2_nearest_to_fixed_cell

    grid2_nearest_to_fixed_cell(
        cell2isfixed, cell2nearest, cell2distance, is_initial, stream_ptr
    )


def grid2_smooth_gauss_seidel(cell2isfix, cell2val, num_iter: int, stream_ptr: int = 0):
    from ..del_msh_dlpack import grid2_smooth_gauss_seidel

    grid2_smooth_gauss_seidel(cell2isfix, cell2val, num_iter, stream_ptr)


def grid2_smooth_gauss_seidel_with_radius(
    cell2isfixed, cell2dist, ratio: float, cell2val, stream_ptr: int = 0
):
    from ..del_msh_dlpack import grid2_smooth_gauss_seidel_with_radius

    grid2_smooth_gauss_seidel_with_radius(
        cell2isfixed, cell2dist, ratio, cell2val, stream_ptr
    )
