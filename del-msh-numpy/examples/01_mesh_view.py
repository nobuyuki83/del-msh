import moderngl
import moderngl_window as mglw
import pyrr
import numpy
import DrawerMesh

class App(mglw.WindowConfig):
    title = "ModernGL: rotating triangle"
    window_size = (800, 600)
    resource_dir = "."
    gl_version = (3, 3)  # macOS でもOK（4,1でも可）
    aspect_ratio = None  # リサイズ時もそのまま

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        V = numpy.array([
            [-0.5, -0.5, 0],
            [+0.5, -0.5, 0],
            [+0, +0.5, 0]], dtype=numpy.float32)
        F = numpy.array([
            [0, 1, 2]], dtype=numpy.uint32)
        E = numpy.array([
            [0, 1],
            [1, 2],
            [2, 0]], dtype=numpy.uint32)
        self.drawer = DrawerMesh.Drawer(
            vtx2xyz=V,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=F, color=(1, 0, 0), mode=moderngl.TRIANGLES),
                DrawerMesh.ElementInfo(index=E, color=(0, 0, 0), mode=moderngl.LINES)]
        )
        self.drawer.init_gl(self.ctx)


    # ★ moderngl_window が呼ぶ描画フック
    def render(self, time: float, frame_time: float):
        self._draw(time)

    # ★ 一部のバックエンドでこちらを見ることがあるので保険
    def on_render(self, time: float, frame_time: float):
        self._draw(time)

    def _draw(self, time: float):
        self.ctx.clear(1.0, 0.8, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        mvp = pyrr.Matrix44.identity()
        self.drawer.paint_gl(mvp)

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)


if __name__ == '__main__':
    mglw.run_window_config(App)
