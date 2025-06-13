#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example
#![allow(unsafe_code)]
#![allow(clippy::undocumented_unsafe_blocks)]

use eframe::{egui, egui_glow, glow};

use crate::PickedObject::{FaceInside, Nothing, VtxInside};
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

struct Geometry {
    vtx2xy_inside: Vec<f32>,
    vtx2xy_outside: Vec<f32>,
}

enum PickedObject {
    Nothing,
    VtxInside(usize),
    FaceInside,
    // VtxOutside(usize),
}

struct MyApp {
    /// Behind an `Arc<Mutex<…>>` so we can pass it to [`egui::PaintCallback`] and paint later.
    drawer_edge: Arc<Mutex<del_glow::drawer_edge2::Drawer>>,
    geometry: Arc<Mutex<Geometry>>,
    is_updated_geometry: bool,
    _gl: Option<Arc<glow::Context>>,
    picked_object: PickedObject,
    penetration: Option<([f32; 2], [f32; 2])>,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let vtx2xy_inside = vec![0.0, 0.0, 0.5, 0.0, 0.5, 0.1, 0.0, 0.1];
        let vtx2xy_outside = del_msh_cpu::polyloop2::from_circle(0.8, 32);
        let geo = Geometry {
            vtx2xy_inside,
            vtx2xy_outside,
        };
        // ---------
        let gl = cc
            .gl
            .as_ref()
            .expect("You need to run eframe with the glow backend");
        let drawer_edge = {
            let mut drawer_mesh = del_glow::drawer_edge2::Drawer::new();
            drawer_mesh.compile_shader(gl);
            drawer_mesh
        };
        Self {
            is_updated_geometry: true,
            picked_object: Nothing,
            drawer_edge: Arc::new(Mutex::new(drawer_edge)),
            geometry: Arc::new(Mutex::new(geo)),
            _gl: Some(gl.clone()),
            penetration: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("ViewRotate: Alt + LeftDrag");
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (id, rect) = ui.allocate_space(ui.available_size());
                self.handle_event(ui, rect, id);
                self.custom_painting(ui, rect);
            });
        });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.drawer_edge.lock().destroy(gl);
        }
    }
}

impl MyApp {
    fn handle_event(&mut self, ui: &mut egui::Ui, rect: egui::Rect, id: egui::Id) {
        self.is_updated_geometry = false;
        let mut geo = self.geometry.lock();
        let ctx = ui.ctx();
        if ctx.input(|i0| {
            i0.pointer.button_pressed(egui::PointerButton::Primary) && i0.modifiers.is_none()
        }) {
            self.picked_object = Nothing;
            if let Some(pos) = ctx.pointer_interact_pos() {
                let pos = pos - rect.left_top();
                let pos_ndc = [
                    2. * pos.x / rect.width() - 1.,
                    1. - 2. * pos.y / rect.height(),
                ];
                if let Some((i_vtx, _pos)) =
                    geo.vtx2xy_inside.chunks(2).enumerate().find(|(_i_vtx, v)| {
                        del_geo_core::edge2::length(&pos_ndc, &[v[0], v[1]]) < 0.01
                    })
                {
                    self.picked_object = VtxInside(i_vtx);
                } else if del_msh_cpu::polyloop2::is_include_a_point(&geo.vtx2xy_inside, &pos_ndc) {
                    self.picked_object = FaceInside;
                }
            }
        }
        // for drag
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());
        if ctx
            .input(|i| i.pointer.button_down(egui::PointerButton::Primary) && i.modifiers.is_none())
        {
            if let Some(pos) = ctx.pointer_interact_pos() {
                let pos1 = pos - rect.left_top();
                let pos1_ndc = [
                    2. * pos1.x / rect.width() - 1.,
                    1. - 2. * pos1.y / rect.height(),
                ];
                let dpos_ndc = [
                    2. * response.drag_motion().x / rect.width(),
                    -2. * response.drag_motion().y / rect.height(),
                ];
                match self.picked_object {
                    VtxInside(i_vtx) => {
                        geo.vtx2xy_inside[i_vtx * 2] = pos1_ndc[0];
                        geo.vtx2xy_inside[i_vtx * 2 + 1] = pos1_ndc[1];
                        self.is_updated_geometry = true;
                    }
                    FaceInside => {
                        for xy in geo.vtx2xy_inside.chunks_mut(2) {
                            xy[0] += dpos_ndc[0];
                            xy[1] += dpos_ndc[1];
                        }
                        self.is_updated_geometry = true;
                    }
                    _ => {}
                }
            }
        }
        if self.is_updated_geometry {
            self.penetration = del_msh_cpu::polyloop2::maximum_penetration_of_included_point2s(
                &geo.vtx2xy_outside,
                &geo.vtx2xy_inside,
            );
        }
    }

    fn custom_painting(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let mat_projection = del_geo_core::mat4_col_major::from_identity();
        let z_flip = del_geo_core::mat4_col_major::from_diagonal(1., 1., -1., 1.);
        let mat_mvp_opengl =
            del_geo_core::mat4_col_major::mult_mat_col_major(&z_flip, &mat_projection);
        let drawer_edge = self.drawer_edge.clone();
        let geo = self.geometry.clone();
        let penetration = self.penetration;
        if penetration.is_some() {
            self.drawer_edge.lock().set_color(&[1.0, 0.0, 0.0]);
        } else {
            self.drawer_edge.lock().set_color(&[0.0, 0.0, 0.0]);
        }
        let callback = egui::PaintCallback {
            rect,
            callback: std::sync::Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                unsafe {
                    gl.clear(glow::DEPTH_BUFFER_BIT);
                    gl.enable(glow::DEPTH_TEST);
                }
                let geo = geo.lock();
                drawer_edge.lock().draw_polyloop2(
                    painter.gl(),
                    &mat_mvp_opengl,
                    &geo.vtx2xy_inside,
                    0.01,
                );
                drawer_edge.lock().draw_polyloop2(
                    painter.gl(),
                    &mat_mvp_opengl,
                    &geo.vtx2xy_outside,
                    0.01,
                );
                if let Some(penetration) = penetration {
                    drawer_edge.lock().draw_edge2(
                        painter.gl(),
                        &mat_mvp_opengl,
                        &penetration.0,
                        &penetration.1,
                        0.01,
                    );
                }
            })),
        };
        ui.painter().add(callback);
    }
}
