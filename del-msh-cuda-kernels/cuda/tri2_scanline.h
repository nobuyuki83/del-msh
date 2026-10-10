#pragma once
#include <stdint.h>

namespace tri2_scanline {

// Iterator over all pixels whose centre lies inside a 2-D triangle.
// Usage (mirrors the Rust TriangleScanlineIter):
//
//   TriangleScanlineIter it(px0, px1, px2);
//   int ix, iy;
//   while (it.next(ix, iy)) { /* use ix, iy */ }
//
struct TriangleScanlineIter {
    float v[3][2];

    int32_t py;
    int32_t y_end;

    int32_t cur_x;
    int32_t x_end;
    bool    has_span;

    __host__ __device__
    TriangleScanlineIter(const float p0[2], const float p1[2], const float p2[2]) {
        v[0][0] = p0[0]; v[0][1] = p0[1];
        v[1][0] = p1[0]; v[1][1] = p1[1];
        v[2][0] = p2[0]; v[2][1] = p2[1];

        float min_y = fminf(fminf(p0[1], p1[1]), p2[1]);
        float max_y = fmaxf(fmaxf(p0[1], p1[1]), p2[1]);

        py    = (int32_t)ceilf(min_y - 0.5f);
        y_end = (int32_t)floorf(max_y - 0.5f);

        cur_x     = 0;
        x_end     = -1;
        has_span  = false;
    }

    // Advance to the next pixel. Returns false when iteration is exhausted.
    __host__ __device__
    bool next(int32_t &out_x, int32_t &out_y) {
        for (;;) {
            if (has_span && cur_x <= x_end) {
                out_x = cur_x++;
                out_y = py;
                return true;
            }
            if (has_span) { ++py; }
            setup_span();
            if (!has_span) { return false; }
        }
    }

private:
    // Returns x-coordinate of edge intersection with horizontal line y,
    // or sets *hit=false if the edge does not cross that y.
    __host__ __device__
    static float edge_intersect(const float p0[2], const float p1[2],
                                float y, bool &hit) {
        float dy = p1[1] - p0[1];
        if (fabsf(dy) < 1e-7f) { hit = false; return 0.f; }
        float ymin = fminf(p0[1], p1[1]);
        float ymax = fmaxf(p0[1], p1[1]);
        if (y < ymin || y >= ymax) { hit = false; return 0.f; }
        float t = (y - p0[1]) / dy;
        hit = true;
        return p0[0] + t * (p1[0] - p0[0]);
    }

    __host__ __device__
    void setup_span() {
        has_span = false;
        while (py <= y_end) {
            float scan_y = (float)py + 0.5f;
            float xs[3];
            int   count = 0;

            bool hit;
            float x;

            x = edge_intersect(v[0], v[1], scan_y, hit); if (hit) xs[count++] = x;
            x = edge_intersect(v[1], v[2], scan_y, hit); if (hit) xs[count++] = x;
            x = edge_intersect(v[2], v[0], scan_y, hit); if (hit) xs[count++] = x;

            if (count >= 2) {
                // Sort up to 3 values (insertion sort, always tiny).
                for (int i = 1; i < count; ++i) {
                    float key = xs[i];
                    int j = i - 1;
                    while (j >= 0 && xs[j] > key) { xs[j+1] = xs[j]; --j; }
                    xs[j+1] = key;
                }
                float x_left  = xs[0];
                float x_right = xs[count - 1];

                int32_t x_start = (int32_t)ceilf(x_left  - 0.5f);
                int32_t x_stop  = (int32_t)floorf(x_right - 0.5f);

                if (x_start <= x_stop) {
                    cur_x    = x_start;
                    x_end    = x_stop;
                    has_span = true;
                    return;
                }
            }
            ++py;
        }
    }
};

} // namespace tri2_scanline
