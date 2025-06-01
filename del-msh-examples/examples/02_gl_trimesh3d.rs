use del_gl_core::gl;

struct MyViewTrg {
    drawer: del_gl_core::drawer_mesh::Drawer,
}

impl del_gl_winit_glutin::viewer3d_for_gl_renderer::GlRenderer for MyViewTrg {
    fn draw(&mut self, gl: &gl::Gl, cam_model: &[f32; 16], cam_projection: &[f32; 16]) {
        self.drawer.draw(gl, cam_model, cam_projection);
    }

    fn initialize(&mut self, gl: &gl::Gl) {
        self.drawer.compile_shader(gl);
        let (tri2vtx, vtx2xyz) = {
            let mut obj = del_msh_cpu::io_obj::WavefrontObj::<usize, f32>::new();
            obj.load("asset/spot/spot_triangulated.obj").unwrap();
            (obj.idx2vtx_xyz, obj.vtx2xyz)
        };
        let edge2vtx = del_msh_cpu::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
        self.drawer.update_vertex(gl, &vtx2xyz, 3);
        self.drawer
            .add_element(gl, gl::TRIANGLES, &tri2vtx, [1.0, 0.0, 0.0]);
        self.drawer
            .add_element(gl, gl::LINES, &edge2vtx, [0.0, 0.0, 0.0]);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let window_attributes = winit::window::Window::default_attributes()
        .with_transparent(false)
        .with_title("02_meshtex")
        .with_inner_size(winit::dpi::PhysicalSize {
            width: 600,
            height: 600,
        });
    let template = glutin::config::ConfigTemplateBuilder::new()
        .with_alpha_size(8)
        .with_transparency(cfg!(cgl_backend));
    let display_builder =
        glutin_winit::DisplayBuilder::new().with_window_attributes(Some(window_attributes));
    let mut app = del_gl_winit_glutin::viewer3d_for_gl_renderer::Viewer3d::new(
        template,
        display_builder,
        Box::new(MyViewTrg {
            drawer: del_gl_core::drawer_mesh::Drawer::new(),
        }),
    );
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.run_app(&mut app)?;
    app.appi.exit_state
}
