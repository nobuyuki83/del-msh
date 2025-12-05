#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example
#![allow(unsafe_code)]
#![allow(clippy::undocumented_unsafe_blocks)]

use eframe::{egui, egui_glow, glow};

use egui::mutex::Mutex;
use glow::HasContext;
use std::sync::Arc;

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([550.0, 600.0]),
        multisampling: 4,
        depth_buffer: 24,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Custom 3D painting in eframe using glow",
        options,
        Box::new(|cc| Ok(Box::new(MyApp::new(cc)))),
    )
}

struct MyApp {
    /// Behind an `Arc<Mutex<…>>` so we can pass it to [`egui::PaintCallback`] and paint later.
    drawer: Arc<Mutex<del_glow::drawer_elem2vtx_vtx2xyz::Drawer>>,
    // mat_modelview: [f32;16],
    mat_projection: [f32; 16],
    trackball: del_geo_core::view_rotation::Trackball<f32>,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tri2vtx, vtx2xyz) = {
            let mut obj = del_msh_cpu::io_wavefront_obj::WavefrontObj::<usize, f32>::new();
            obj.load("asset/spot/spot_triangulated.obj").unwrap();
            (obj.idx2vtx_xyz, obj.vtx2xyz)
        };
        let edge2vtx = del_msh_cpu::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
        //
        let gl = cc
            .gl
            .as_ref()
            .expect("You need to run eframe with the glow backend");
        let mut drawer = del_glow::drawer_elem2vtx_vtx2xyz::Drawer::new();
        drawer.compile_shader(gl);
        drawer.set_vtx2xyz(gl, &vtx2xyz, 3);
        drawer.add_elem2vtx(gl, glow::LINES, &edge2vtx, [0.0, 0.0, 0.0]);
        drawer.add_elem2vtx(gl, glow::TRIANGLES, &tri2vtx, [1.0, 0.8, 0.8]);
        Self {
            drawer: Arc::new(Mutex::new(drawer)),
            // mat_modelview: del_geo_core::mat4_col_major::from_identity(),
            trackball: del_geo_core::view_rotation::Trackball::default(),
            mat_projection: del_geo_core::mat4_col_major::from_identity(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Rotate: Alt + LeftDrag");
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (id, rect) = ui.allocate_space(ui.available_size());
                self.handle_event(ui, rect, id);
                self.custom_painting(ui, rect);
            });
        });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.drawer.lock().destroy(gl);
        }
    }
}

impl MyApp {
    fn handle_event(&mut self, ui: &mut egui::Ui, rect: egui::Rect, id: egui::Id) {
        let response = ui.interact(rect, id, egui::Sense::drag());
        let ctx = ui.ctx();
        if ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary) && i.modifiers.alt) {
            let xy = response.drag_motion();
            let dx = 2.0 * xy.x / rect.width();
            let dy = -2.0 * xy.y / rect.height();
            self.trackball.camera_rotation(dx, dy);
        }
    }
    fn custom_painting(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        // Clone locals so we can move them into the paint callback:
        let mat_modelview = self.trackball.mat4_col_major();
        let mat_projection = self.mat_projection;
        let z_flip = del_geo_core::mat4_col_major::from_diagonal(1., 1., -1., 1.);
        let mat_projection_for_opengl =
            del_geo_core::mat4_col_major::mult_mat_col_major(&z_flip, &mat_projection);
        let drawer = self.drawer.clone();
        let callback = egui::PaintCallback {
            rect,
            callback: std::sync::Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                unsafe {
                    gl.clear(glow::DEPTH_BUFFER_BIT);
                    gl.enable(glow::DEPTH_TEST);
                }
                drawer
                    .lock()
                    .draw(painter.gl(), &mat_modelview, &mat_projection_for_opengl);
            })),
        };
        ui.painter().add(callback);
    }
}
