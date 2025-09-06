import numpy as np
import moderngl
import moderngl_window as mglw
import DrawerMeshVtxColor
import pyrr

class App(mglw.WindowConfig):
    title = "ModernGL: rotating triangle"
    window_size = (800, 600)
    resource_dir = "."
    gl_version = (3, 3)  # macOS でもOK（4,1でも可）
    aspect_ratio = None  # リサイズ時もそのまま

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        V = np.array([
            [-0.5, -0.5, 0],
            [+0.5, -0.5, 0],
            [+0, +0.5, 0]], dtype=np.float32)
        C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]], dtype=np.float32)
        F = np.array([
            [0, 1, 2]], dtype=np.uint32)
        self.drawer = DrawerMeshVtxColor.Drawer(V=V.tobytes(), C=C.tobytes(), F=F.tobytes())
        self.drawer.init_gl(self.ctx)

    # ★ moderngl_window が呼ぶ描画フック
    def render(self, time: float, frame_time: float):
        self._draw(time)

    # ★ 一部のバックエンドでこちらを見ることがあるので保険
    def on_render(self, time: float, frame_time: float):
        self._draw(time)

    def _draw(self, t: float):
        self.ctx.clear(1.0, 0.8, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        mvp = pyrr.Matrix44.identity()
        self.drawer.paint_gl(mvp)

if __name__ == "__main__":
    mglw.run_window_config(App)
