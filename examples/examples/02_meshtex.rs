use del_gl_core::gl;
use glutin::display::GetGlDisplay;

pub struct MyApp {
    pub appi: del_gl_winit_glutin::app_internal::AppInternal,
    pub renderer: Option<del_gl_core::drawer_mesh::Drawer>,
    pub view_rot: del_geo_core::view_rotation::Trackball,
    pub view_prj: del_geo_core::view_projection::Perspective,
    pub ui_state: del_gl_core::view_ui_state::UiState,
    pub is_left_btn_down_not_for_view_ctrl: bool,
    pub is_view_changed: bool,
}

impl MyApp {
    pub fn new(
        template: glutin::config::ConfigTemplateBuilder,
        display_builder: glutin_winit::DisplayBuilder,
    ) -> Self {
        Self {
            appi: del_gl_winit_glutin::app_internal::AppInternal::new(template, display_builder),
            renderer: None,
            ui_state: del_gl_core::view_ui_state::UiState::new(),
            view_rot: del_geo_core::view_rotation::Trackball::new(),
            view_prj: del_geo_core::view_projection::Perspective {
                lens: 24.,
                near: 0.5,
                far: 3.0,
                cam_pos: [0., 0., 2.],
                proj_direction: false,
                scale: 1.,
            },
            is_left_btn_down_not_for_view_ctrl: false,
            is_view_changed: false,
        }
    }
}

impl winit::application::ApplicationHandler for MyApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Some(app_state) = self.appi.resumed(event_loop) else {
            return;
        };
        // The context needs to be current for the Renderer to set up shaders and
        // buffers. It also performs function loading, which needs a current context on
        // WGL.
        use glutin::display::GetGlDisplay;
        self.renderer.get_or_insert_with(|| {
            let gl_display = &app_state.gl_context.display();
            let gl = del_gl_core::gl::Gl::load_with(|symbol| {
                let symbol = std::ffi::CString::new(symbol).unwrap();
                use glutin::display::GlDisplay;
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            });
            unsafe {
                gl.Enable(gl::DEPTH_TEST);
            }
            let mut drawer = del_gl_core::drawer_mesh::Drawer::new();
            drawer.compile_shader(&gl);
            let (tri2vtx, vtx2xyz) = {
                let mut obj = del_msh_core::io_obj::WavefrontObj::<usize, f32>::new();
                obj.load("asset/spot/spot_triangulated.obj").unwrap();
                (obj.idx2vtx_xyz, obj.vtx2xyz)
            };
            let edge2vtx = del_msh_core::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
            drawer.update_vertex(&gl, &vtx2xyz, 3);
            drawer.add_element(&gl, gl::TRIANGLES, &tri2vtx, [1.0, 0.0, 0.0]);
            drawer.add_element(&gl, gl::LINES, &edge2vtx, [0.0, 0.0, 0.0]);
            drawer
        });
        assert!(self.appi.state.replace(app_state).is_none());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use glutin::prelude::GlSurface;
        self.is_left_btn_down_not_for_view_ctrl = false;
        match event {
            winit::event::WindowEvent::Resized(size) if size.width != 0 && size.height != 0 => {
                // Some platforms like EGL require resizing GL surface to update the size
                // Notable platforms here are Wayland and macOS, other don't require it
                // and the function is no-op, but it's wise to resize it for portability
                // reasons.
                if let Some(del_gl_winit_glutin::app_internal::AppState {
                    gl_context,
                    gl_surface,
                    window: _,
                }) = self.appi.state.as_ref()
                {
                    gl_surface.resize(
                        gl_context,
                        std::num::NonZeroU32::new(size.width).unwrap(),
                        std::num::NonZeroU32::new(size.height).unwrap(),
                    );
                    // let renderer = self.renderer.as_ref().unwrap();
                    // renderer.resize(size.width as i32, size.height as i32);
                    let gl_display = &gl_context.display();
                    let gl = del_gl_core::gl::Gl::load_with(|symbol| {
                        let symbol = std::ffi::CString::new(symbol).unwrap();
                        use glutin::display::GlDisplay;
                        gl_display.get_proc_address(symbol.as_c_str()).cast()
                    });
                    unsafe {
                        gl.Viewport(0, 0, size.width as i32, size.height as i32);
                    }
                }
            }
            winit::event::WindowEvent::CloseRequested
            | winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            _ => (),
        }
        let redraw = del_gl_winit_glutin::view_navigation(
            event,
            &mut self.ui_state,
            &mut self.view_prj,
            &mut self.view_rot,
        );
        if redraw {
            if let Some(state) = &self.appi.state {
                state.window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        use glutin::prelude::GlSurface;
        if let Some(del_gl_winit_glutin::app_internal::AppState {
            gl_context,
            gl_surface,
            window,
        }) = self.appi.state.as_ref()
        {
            let img_shape = { (window.inner_size().width, window.inner_size().height) };
            let cam_model = self.view_rot.mat4_col_major();
            let cam_projection = self
                .view_prj
                .mat4_col_major(img_shape.0 as f32 / img_shape.1 as f32);
            let renderer = self.renderer.as_ref().unwrap();
            use glutin::display::GetGlDisplay;
            let gl_display = &gl_context.display();
            let gl = del_gl_core::gl::Gl::load_with(|symbol| {
                let symbol = std::ffi::CString::new(symbol).unwrap();
                use glutin::display::GlDisplay;
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            });
            unsafe {
                gl.ClearColor(0.3, 0.3, 0.3, 1.0);
                gl.Clear(gl::COLOR_BUFFER_BIT);
                gl.Clear(gl::DEPTH_BUFFER_BIT);
            }
            renderer.draw(&gl, &cam_model, &cam_projection);
            window.request_redraw();
            gl_surface.swap_buffers(gl_context).unwrap();
        }
    }

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // This event is only raised on Android, where the backing NativeWindow for a GL
        // Surface can appear and disappear at any moment.
        println!("Android window removed");
        self.appi.suspended();
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
    let mut app = MyApp::new(template, display_builder);
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.run_app(&mut app)?;
    app.appi.exit_state
}
