#include <cuda/std/tuple>
#include <cuda/std/array>
#include <optional>


class DDA {
public:
  float x, y;
  float slope_x, slope_y;
  float width_f, height_f;
  const cuda::std::array<float,2> p0;
  const cuda::std::array<float,2> p1;
public:
  __device__
  DDA(
      const float* p0_,
      const float* p1_,
      uint32_t img_width,
      uint32_t img_height):
      p0 {p0_[0], p0_[1]},
      p1 {p1_[0], p1_[1]},
      x(p0_[0]),
      y(p0_[1])
  {
    this->width_f = float(img_width);
    this->height_f = float(img_height);
    const float dx = p1_[0] - p0_[0];
    const float dy = p1_[1] - p0_[1];
    const float step =  max(abs(dx), abs(dy));
    this->slope_y = dy / step;
    this->slope_x = dx / step;
  }

  struct PixelCoord {
    int32_t ix;
    int32_t iy;
  };


  __device__
  auto pixel() -> PixelCoord const {
    const int32_t ix = int32_t(floor(x));
    const int32_t iy = int32_t(floor(y));
    //const int32_t ix = x;
    //const int32_t iy = y;
    return PixelCoord{ix, iy};
  }

  __device__
  bool is_valid() const {
      float eps = 1.0e-3f;
//      if( x < min(p0[0], p1[0])-eps  || x > max(p0[0], p1[0])+eps ){ return false; }
//      if( y < min(p0[1], p1[1])-eps  || y > max(p0[1], p1[1])+eps ){ return false; }
      if( abs(x - p0[0]) > abs(p1[0] - p0[0])+eps ){ return false; }
      if( abs(y - p0[1]) > abs(p1[1] - p0[1])+eps ){ return false; }
      return true;
  }

  __device__
  void move() {
      x = x + slope_x;
      y = y + slope_y;
  }
};