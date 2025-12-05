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
    drawer_mesh: Arc<Mutex<del_glow::drawer_elem2vtx_vtx2xyz::Drawer>>,
    drawer_sphere: Arc<Mutex<del_glow::drawer_elem2vtx_vtx2xyz::Drawer>>,
    mat_projection: [f32; 16],
    trackball: del_geo_core::view_rotation::Trackball<f32>,
    tri2vtx: Vec<u32>,
    vtx2xyz: Vec<f32>,
    picked_tri: Option<(usize, [f32; 3])>,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc
            .gl
            .as_ref()
            .expect("You need to run eframe with the glow backend");
        let (tri2vtx, vtx2xyz) = {
            let mut obj = del_msh_cpu::io_wavefront_obj::WavefrontObj::<u32, f32>::new();
            obj.load("asset/spot/spot_triangulated.obj").unwrap();
            (obj.idx2vtx_xyz, obj.vtx2xyz)
        };
        let drawer_mesh = {
            let mut drawer_mesh = del_glow::drawer_elem2vtx_vtx2xyz::Drawer::new();
            drawer_mesh.compile_shader(gl);
            let edge2vtx = del_msh_cpu::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
            drawer_mesh.set_vtx2xyz(gl, &vtx2xyz, 3);
            drawer_mesh.add_elem2vtx(gl, glow::LINES, &edge2vtx, [0.0, 0.0, 0.0]);
            drawer_mesh.add_elem2vtx(gl, glow::TRIANGLES, &tri2vtx, [0.8, 0.8, 0.9]);
            drawer_mesh
        };
        let drawer_sphere = {
            let mut drawer_sphere = del_glow::drawer_elem2vtx_vtx2xyz::Drawer::new();
            drawer_sphere.compile_shader(gl);
            let (tri2vtx, vtx2xyz) =
                del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(0.1, 32, 32);
            drawer_sphere.set_vtx2xyz(gl, &vtx2xyz, 3);
            drawer_sphere.add_elem2vtx(gl, glow::TRIANGLES, &tri2vtx, [1.0, 0.5, 0.5]);
            drawer_sphere
        };
        Self {
            drawer_mesh: Arc::new(Mutex::new(drawer_mesh)),
            drawer_sphere: Arc::new(Mutex::new(drawer_sphere)),
            trackball: del_geo_core::view_rotation::Trackball::default(),
            mat_projection: del_geo_core::mat4_col_major::from_identity(),
            tri2vtx: tri2vtx.to_vec(),
            vtx2xyz,
            picked_tri: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("View Rot: Alt + LeftDrag");
            ui.label("Pick: LeftPress");
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (id, rect) = ui.allocate_space(ui.available_size());
                self.handle_event(ui, rect, id);
                self.custom_painting(ui, rect);
            });
        });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.drawer_mesh.lock().destroy(gl);
        }
    }
}

impl MyApp {
    fn picking_ray(&self, pos: egui::Pos2, rect: egui::Rect) -> ([f32; 3], [f32; 3]) {
        let mat_modelview = self.trackball.mat4_col_major();
        let mat_projection = self.mat_projection;
        let transform_world2ndc =
            del_geo_core::mat4_col_major::mult_mat_col_major(&mat_projection, &mat_modelview);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
        let pos = pos - rect.left_top();
        let ndc_x = 2. * pos.x / rect.width() - 1.;
        let ndc_y = 1. - 2. * pos.y / rect.height();
        let world_stt = del_geo_core::mat4_col_major::transform_homogeneous(
            &transform_ndc2world,
            &[ndc_x, ndc_y, 1.],
        )
        .unwrap();
        let world_end = del_geo_core::mat4_col_major::transform_homogeneous(
            &transform_ndc2world,
            &[ndc_x, ndc_y, -1.],
        )
        .unwrap();
        let ray_org = world_stt;
        let ray_dir = del_geo_core::vec3::sub(&world_end, &world_stt);
        (ray_org, ray_dir)
    }

    fn handle_event(&mut self, ui: &mut egui::Ui, rect: egui::Rect, id: egui::Id) {
        let response = ui.interact(rect, id, egui::Sense::drag());
        let ctx = ui.ctx();
        if ctx.input(|i| {
            i.pointer.button_pressed(egui::PointerButton::Primary) && i.modifiers.is_none()
        }) {
            self.picked_tri = if let Some(pos) = ctx.pointer_interact_pos() {
                let (ray_org, ray_dir) = self.picking_ray(pos, rect);
                let res = del_msh_cpu::trimesh3_search_bruteforce::first_intersection_ray(
                    &ray_org,
                    &ray_dir,
                    &self.tri2vtx,
                    &self.vtx2xyz,
                );
                if let Some((depth, i_tri)) = res {
                    let pos = del_geo_core::vec3::axpy(depth, &ray_dir, &ray_org);
                    Some((i_tri as usize, pos))
                } else {
                    None
                }
            } else {
                None
            };
        }
        if ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary) && i.modifiers.alt) {
            let xy = response.drag_motion();
            let dx = 2.0 * xy.x / rect.width();
            let dy = -2.0 * xy.y / rect.height();
            self.trackball.camera_rotation(dx, dy);
        }
    }

    fn custom_painting(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let mat_modelview = self.trackball.mat4_col_major();
        let mat_projection = self.mat_projection;
        let z_flip = del_geo_core::mat4_col_major::from_diagonal(1., 1., -1., 1.);
        let mat_projection_for_opengl =
            del_geo_core::mat4_col_major::mult_mat_col_major(&z_flip, &mat_projection);
        // del_geo_core::view_rotation::Trackball::new();
        let drawer_mesh = self.drawer_mesh.clone();
        let drawer_sphere = self.drawer_sphere.clone();
        let mat_modelview_for_sphere = if let Some((_i_tri, pos)) = self.picked_tri {
            let mat = del_geo_core::mat4_col_major::mult_mat_col_major(
                &mat_modelview,
                &del_geo_core::mat4_col_major::from_translate(&pos),
            );
            Some(mat)
        } else {
            None
        };
        let callback = egui::PaintCallback {
            rect,
            callback: std::sync::Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                unsafe {
                    gl.clear(glow::DEPTH_BUFFER_BIT);
                    gl.enable(glow::DEPTH_TEST);
                }
                drawer_mesh
                    .lock()
                    .draw(painter.gl(), &mat_modelview, &mat_projection_for_opengl);
                if let Some(mat_modelview_for_sphere) = mat_modelview_for_sphere {
                    drawer_sphere.lock().draw(
                        painter.gl(),
                        &mat_modelview_for_sphere,
                        &mat_projection_for_opengl,
                    );
                }
            })),
        };
        ui.painter().add(callback);
    }
}
