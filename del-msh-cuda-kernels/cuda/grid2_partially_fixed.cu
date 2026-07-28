#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

extern "C" {

constexpr uint32_t INVALID_CELL = UINT32_MAX;

__device__ __forceinline__
float squared_distance(
    uint32_t i_cell,
    uint32_t i_seed,
    uint32_t width)
{
    const int x0 = static_cast<int>(i_cell % width);
    const int y0 = static_cast<int>(i_cell / width);
    const int x1 = static_cast<int>(i_seed % width);
    const int y1 = static_cast<int>(i_seed / width);

    const float dx = static_cast<float>(x0 - x1);
    const float dy = static_cast<float>(y0 - y1);

    return dx * dx + dy * dy;
}



/// Initialize the nearest-cell map.
///
/// Fixed cells point to themselves.
/// Non-fixed cells are initialized to INVALID_CELL.
///
/// cell2isfixed: length = width * height
/// cell2nearest: length = width * height
__global__ void nearest_to_fixed_cell_initialize(
    uint32_t width,
    uint32_t height,
    const uint8_t* __restrict__ cell2isfixed,
    uint32_t* __restrict__ cell2nearest)
{
    const uint32_t i_cell =
        blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t num_cells = width * height;

    if (i_cell >= num_cells) {
        return;
    }

    cell2nearest[i_cell] =
        cell2isfixed[i_cell] != 0
        ? i_cell
        : INVALID_CELL;
}


/// One Jump Flooding pass.
///
/// src_cell2nearest and dst_cell2nearest must point to different arrays.
///
/// Each cell checks the current candidate and candidates located at
/// offsets {-step, 0, +step} in x and y.
__global__ void nearest_to_fixed_cell_jump_flood(
    uint32_t width,
    uint32_t height,
    const uint32_t* __restrict__ src_cell2nearest,
    uint32_t* __restrict__ dst_cell2nearest,
    uint32_t step)
{
    const uint32_t i_cell =
        blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t num_cells = width * height;

    if (i_cell >= num_cells) {
        return;
    }

    const int x = static_cast<int>(i_cell % width);
    const int y = static_cast<int>(i_cell / width);

    uint32_t best_seed = src_cell2nearest[i_cell];
    float best_distance_squared =
        best_seed == INVALID_CELL
        ? FLT_MAX
        : squared_distance(i_cell, best_seed, width);

    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            if (offset_x == 0 && offset_y == 0) {
                continue;
            }

            const int candidate_x =
                x + offset_x * static_cast<int>(step);
            const int candidate_y =
                y + offset_y * static_cast<int>(step);

            if (
                candidate_x < 0 ||
                candidate_x >= static_cast<int>(width) ||
                candidate_y < 0 ||
                candidate_y >= static_cast<int>(height))
            {
                continue;
            }

            const uint32_t candidate_cell =
                static_cast<uint32_t>(candidate_y) * width
                + static_cast<uint32_t>(candidate_x);

            const uint32_t candidate_seed =
                src_cell2nearest[candidate_cell];

            if (candidate_seed == INVALID_CELL) {
                continue;
            }

            const float candidate_distance_squared =
                squared_distance(i_cell, candidate_seed, width);

            if (
                candidate_distance_squared < best_distance_squared ||
                (
                    candidate_distance_squared == best_distance_squared &&
                    candidate_seed < best_seed
                ))
            {
                best_seed = candidate_seed;
                best_distance_squared = candidate_distance_squared;
            }
        }
    }

    dst_cell2nearest[i_cell] = best_seed;
}


/// Convert the nearest-cell indices into Euclidean distances.
///
/// The distance is measured in grid-cell units.
__global__ void nearest_to_fixed_cell_compute_distance(
    uint32_t width,
    uint32_t height,
    const uint32_t* __restrict__ cell2nearest,
    float* __restrict__ cell2distance)
{
    const uint32_t i_cell =
        blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t num_cells = width * height;

    if (i_cell >= num_cells) {
        return;
    }

    const uint32_t i_seed = cell2nearest[i_cell];

    if (i_seed == INVALID_CELL) {
        cell2distance[i_cell] = FLT_MAX;
        return;
    }

    cell2distance[i_cell] =
        sqrtf(squared_distance(i_cell, i_seed, width));
}


/// Red-black Gauss-Seidel smoothing pass (in-place).
///
/// The grid is partitioned into "red" cells (iw+ih even) and "black" cells
/// (iw+ih odd).  Each kernel launch updates only the cells whose colour
/// matches `color` (0 = red, 1 = black).  Because the 4-connected neighbours
/// of a red cell are always black and vice-versa, the two sets are
/// independent within each pass, so the update is race-free in-place.
///
/// One full Gauss-Seidel iteration = two kernel launches: color=0 then color=1.
///
/// cell2isfix : (height*width,)  u8   – 1 = fixed (skipped), 0 = free
/// cell2val   : (height*width*num_vdim,)  f32  – values, updated in-place
__global__
void smooth_red_black_gauss_seidel(
    uint32_t grid_w,
    uint32_t grid_h,
    const uint8_t* __restrict__ cell2isfix,
    uint32_t num_vdim,
    float* __restrict__ cell2val,
    uint32_t color)
{
    const uint32_t i_cell = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_cell = grid_h * grid_w;
    if (i_cell >= num_cell) { return; }

    const uint32_t iw = i_cell % grid_w;
    const uint32_t ih = i_cell / grid_w;
    if (((iw + ih) & 1u) != color) { return; }   // skip wrong colour
    if (cell2isfix[i_cell] != 0) { return; }       // skip fixed cells

    // Count valid neighbours and accumulate per-dimension sums.
    uint32_t n_sum = 0u;
    if (ih > 0u)            ++n_sum;
    if (ih < grid_h - 1u)  ++n_sum;
    if (iw > 0u)            ++n_sum;
    if (iw < grid_w - 1u)  ++n_sum;
    if (n_sum == 0u) { return; }

    const float inv_n = 1.0f / static_cast<float>(n_sum);

    for (uint32_t i_vdim = 0u; i_vdim < num_vdim; ++i_vdim) {
        float v_sum = 0.0f;
        if (ih > 0u)           v_sum += cell2val[((ih - 1u) * grid_w + iw) * num_vdim + i_vdim];
        if (ih < grid_h - 1u)  v_sum += cell2val[((ih + 1u) * grid_w + iw) * num_vdim + i_vdim];
        if (iw > 0u)           v_sum += cell2val[(ih * grid_w + iw - 1u) * num_vdim + i_vdim];
        if (iw < grid_w - 1u)  v_sum += cell2val[(ih * grid_w + iw + 1u) * num_vdim + i_vdim];
        cell2val[i_cell * num_vdim + i_vdim] = v_sum * inv_n;
    }
}


/// Jacobi smoothing with a per-cell radius (parallel ping-pong version).
///
/// The neighbourhood offset r = floor(cell2dist[i] * ratio).  Each thread
/// reads exclusively from cell2val_pre and writes to cell2val_pos, so all
/// updates are race-free regardless of the radius.
///
/// Fixed cells copy their value unchanged: cell2val_pos[i] = cell2val_pre[i].
///
/// One call = one Jacobi step.  The caller is responsible for ping-pong
/// (swapping pre/pos between calls) or copying pos back to the working buffer.
///
/// cell2isfixed : (h*w,)          u8  – 1 = fixed, 0 = free
/// cell2dist    : (h*w,)          f32 – distance to nearest seed (in cells)
/// cell2val_pre : (h*w*num_vdim,) f32 – read-only input
/// cell2val_pos : (h*w*num_vdim,) f32 – write-only output
__global__
void smooth_gauss_seidel_with_radius(
    uint32_t grid_w,
    uint32_t grid_h,
    const uint8_t* __restrict__ cell2isfixed,
    const float*  __restrict__ cell2dist,
    float ratio,
    uint32_t num_vdim,
    const float* __restrict__ cell2val_pre,
    float*       __restrict__ cell2val_pos)
{
    const uint32_t i_cell = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_cell = grid_w * grid_h;
    if (i_cell >= num_cell) { return; }

    // Fixed cells: preserve value as-is
    if (cell2isfixed[i_cell] != 0u) {
        for (uint32_t i_vdim = 0u; i_vdim < num_vdim; ++i_vdim) {
            cell2val_pos[i_cell * num_vdim + i_vdim] =
                cell2val_pre[i_cell * num_vdim + i_vdim];
        }
        return;
    }

    const int iw1 = static_cast<int>(i_cell % grid_w);
    const int ih1 = static_cast<int>(i_cell / grid_w);
    const int r   = static_cast<int>(floorf(cell2dist[i_cell] * ratio));

    const int ih0 = ih1 - r;
    const int ih2 = ih1 + r;
    const int iw0 = iw1 - r;
    const int iw2 = iw1 + r;

    uint32_t n_sum = 0u;
    if (ih0 >= 0)                       ++n_sum;
    if (ih2 < static_cast<int>(grid_h)) ++n_sum;
    if (iw0 >= 0)                       ++n_sum;
    if (iw2 < static_cast<int>(grid_w)) ++n_sum;

    if (n_sum == 0u) {
        // Isolated cell: keep current value
        for (uint32_t i_vdim = 0u; i_vdim < num_vdim; ++i_vdim) {
            cell2val_pos[i_cell * num_vdim + i_vdim] =
                cell2val_pre[i_cell * num_vdim + i_vdim];
        }
        return;
    }

    const float inv_n = 1.0f / static_cast<float>(n_sum);

    for (uint32_t i_vdim = 0u; i_vdim < num_vdim; ++i_vdim) {
        float v_sum = 0.0f;
        if (ih0 >= 0)
            v_sum += cell2val_pre[(static_cast<uint32_t>(ih0) * grid_w
                                   + static_cast<uint32_t>(iw1)) * num_vdim + i_vdim];
        if (ih2 < static_cast<int>(grid_h))
            v_sum += cell2val_pre[(static_cast<uint32_t>(ih2) * grid_w
                                   + static_cast<uint32_t>(iw1)) * num_vdim + i_vdim];
        if (iw0 >= 0)
            v_sum += cell2val_pre[(static_cast<uint32_t>(ih1) * grid_w
                                   + static_cast<uint32_t>(iw0)) * num_vdim + i_vdim];
        if (iw2 < static_cast<int>(grid_w))
            v_sum += cell2val_pre[(static_cast<uint32_t>(ih1) * grid_w
                                   + static_cast<uint32_t>(iw2)) * num_vdim + i_vdim];
        cell2val_pos[i_cell * num_vdim + i_vdim] = v_sum * inv_n;
    }
}


}