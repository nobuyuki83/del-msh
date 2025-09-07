import moderngl
import moderngl_window as mglw
import pyrr
import numpy as np
import DrawerMesh
import view_navigation3
from del_msh_numpy import TriMesh

class App(mglw.WindowConfig):
    title = "ModernGL: rotating triangle"
    window_size = (800, 600)
    resource_dir = "."
    gl_version = (3, 3)  # macOS でもOK（4,1でも可）
    aspect_ratio = None  # リサイズ時もそのまま
    nav = view_navigation3.ViewNavigation3()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drawer.init_gl(self.ctx)

    def view_transformation_matrix_for_gl(self):
        proj = self.nav.projection_matrix()
        modelview = self.nav.modelview_matrix()
        zinv = pyrr.Matrix44(value=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1), dtype=np.float32)
        return zinv * proj * modelview

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 0.8, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.polygon_offset = 1.1, 4.0
        mvp = self.view_transformation_matrix_for_gl()
        self.drawer.paint_gl(mvp)

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self.nav.update_cursor_position(x, y)
        if button == 1:
            self.nav.btn_left = True

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        if button == 1:
            self.nav.btn_left = False

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.nav.update_cursor_position(x, y)
        if self.nav.btn_left:
            if self.wnd.modifiers.alt:
                self.nav.camera_rotation()
                self.dirty = True

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        pass

    def on_resize(self, width: int, height: int) -> None:
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)
        self.nav.win_width = width
        self.nav.win_height = height


def draw_mesh(tri2vtx, vtx2xyz):
    edge2vtx = TriMesh.edge2vtx(tri2vtx=tri2vtx, num_vtx=vtx2xyz.shape[0])
    drawer = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 0, 0), mode=moderngl.TRIANGLES),
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES)]
    )
    app = App
    app.drawer = drawer
    app.nav.view_height = 1.3
    mglw.run_window_config(app)


if __name__ == "__main__":
    '''
    from util_moderngl_qt import primitive_shapes
    draw_mesh(*primitive_shapes.sphere(radius=1.0, n_longtitude=8, n_latitude=16))
    draw_mesh(*primitive_shapes.cylinder(radius=1.0))
    '''
    #
    draw_mesh(*TriMesh.hemisphere(radius=1.0, ndiv_longtitude=8))
    draw_mesh(*TriMesh.torus(major_radius=0.4, minor_radius=0.2))
    draw_mesh(*TriMesh.capsule(radius=0.1, height=1.2, ndiv_longtitude=8))
    draw_mesh(*TriMesh.cylinder(radius=0.3, height=1.2))
    draw_mesh(*TriMesh.sphere(radius=1.0))
