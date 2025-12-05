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
    drawer_edge: Arc<Mutex<del_glow::drawer_elem2vtx_vtx2xyz::Drawer>>,
    drawer_tri: Arc<Mutex<del_glow::drawer_tri2node2xyz_tri2node2rgb::Drawer>>,
    // mat_modelview: [f32;16],
    mat_projection: [f32; 16],
    trackball: del_geo_core::view_rotation::Trackball<f32>,
    tri2vtx: Vec<u32>,
    vtx2xyz: Vec<f32>,
    tri2dist: Vec<usize>,
    tri2flag: Vec<u8>,
    tri2tri: Vec<u32>,
    is_updated_tri2flag: bool,
    gl: Option<Arc<glow::Context>>,
    cur_dist: Option<usize>,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tri2vtx, vtx2xyz) = {
            let mut obj = del_msh_cpu::io_wavefront_obj::WavefrontObj::<u32, f32>::new();
            obj.load("asset/spot/spot_triangulated.obj").unwrap();
            (obj.idx2vtx_xyz, obj.vtx2xyz)
        };
        let num_tri = tri2vtx.len() / 3;
        let tri2node2xyz =
            del_msh_cpu::unindex::unidex_vertex_attribute_for_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let tri2tri = del_msh_cpu::elem2elem::from_uniform_mesh(
            &tri2vtx,
            3,
            &[0, 2, 4, 6],
            &[1, 2, 2, 0, 0, 1],
            vtx2xyz.len() / 3,
        );
        assert_eq!(tri2node2xyz.len(), num_tri * 9);
        // ---------
        let gl = cc
            .gl
            .as_ref()
            .expect("You need to run eframe with the glow backend");
        let drawer_edge = {
            let mut drawer_mesh = del_glow::drawer_elem2vtx_vtx2xyz::Drawer::new();
            drawer_mesh.compile_shader(gl);
            let edge2vtx = del_msh_cpu::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
            drawer_mesh.set_vtx2xyz(gl, &vtx2xyz, 3);
            drawer_mesh.add_elem2vtx(gl, glow::LINES, &edge2vtx, [0.0, 0.0, 0.0]);
            // drawer_mesh.add_element(&gl, glow::TRIANGLES, &tri2vtx, [0.8, 0.8, 0.9]);
            drawer_mesh
        };
        let drawer_tri = {
            let mut drawer_tri = del_glow::drawer_tri2node2xyz_tri2node2rgb::Drawer::new();
            drawer_tri.compile_shader(gl);
            drawer_tri.update_tri2node2xyz(gl, &tri2node2xyz);
            let tri2node2rgb = vec![0.9; num_tri * 9];
            drawer_tri.update_tri2node2rgb(gl, &tri2node2rgb);
            drawer_tri
        };
        Self {
            drawer_edge: Arc::new(Mutex::new(drawer_edge)),
            drawer_tri: Arc::new(Mutex::new(drawer_tri)),
            trackball: del_geo_core::view_rotation::Trackball::default(),
            mat_projection: del_geo_core::mat4_col_major::from_identity(),
            tri2vtx,
            vtx2xyz,
            tri2tri,
            tri2dist: vec![0; num_tri],
            tri2flag: vec![0; num_tri],
            is_updated_tri2flag: true,
            gl: Some(gl.clone()),
            cur_dist: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("ViewRotate: Alt + LeftDrag");
            if ui.button("clear").clicked() {
                self.tri2flag.iter_mut().for_each(|v| *v = 0);
                self.is_updated_tri2flag = true;
            }
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
        let ctx = ui.ctx();
        if ctx.input(|i0| {
            i0.pointer.button_pressed(egui::PointerButton::Primary) && i0.modifiers.is_none()
        }) {
            // mouse pressed
            let num_tri = self.tri2dist.len();
            if let Some(pos) = ctx.pointer_interact_pos() {
                let (ray_org, ray_dir) = self.picking_ray(pos, rect);
                if let Some((_depth, i_tri)) =
                    del_msh_cpu::trimesh3_search_bruteforce::first_intersection_ray(
                        &ray_org,
                        &ray_dir,
                        &self.tri2vtx,
                        &self.vtx2xyz,
                    )
                {
                    // hit something
                    self.tri2flag[i_tri as usize] = 1;
                    self.tri2dist = del_msh_cpu::dijkstra::elem2dist_for_uniform_mesh(
                        i_tri as usize,
                        &self.tri2tri,
                        num_tri,
                    );
                    self.cur_dist = Some(0);
                    self.is_updated_tri2flag = true;
                }
            }
        }
        if ctx.input(|i0| {
            i0.pointer.button_released(egui::PointerButton::Primary) && i0.modifiers.is_none()
        }) {
            if let Some(cur_dist) = self.cur_dist {
                self.tri2flag
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i_tri, flg)| {
                        if self.tri2dist[i_tri] <= cur_dist {
                            *flg = 1
                        } else {
                            // *flg = 0
                        }
                    });
            }
            self.cur_dist = None;
            self.is_updated_tri2flag = true;
        }
        if ctx
            .input(|i| i.pointer.button_down(egui::PointerButton::Primary) && i.modifiers.is_none())
        {
            if let Some(pos) = ctx.pointer_interact_pos() {
                let (ray_org, ray_dir) = self.picking_ray(pos, rect);
                if let Some((_depth, i_tri)) =
                    del_msh_cpu::trimesh3_search_bruteforce::first_intersection_ray(
                        &ray_org,
                        &ray_dir,
                        &self.tri2vtx,
                        &self.vtx2xyz,
                    )
                {
                    self.cur_dist = Some(self.tri2dist[i_tri as usize]);
                    self.is_updated_tri2flag = true;
                }
            }
        }
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());
        if ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary) && i.modifiers.alt) {
            let xy = response.drag_motion();
            let dx = 2.0 * xy.x / rect.width();
            let dy = -2.0 * xy.y / rect.height();
            self.trackball.camera_rotation(dx, dy);
        }
    }
    fn custom_painting(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        if self.is_updated_tri2flag {
            let mut tri2node2rgb: Vec<f32> = self
                .tri2flag
                .iter()
                .flat_map(|&flag| {
                    if flag == 1 {
                        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                    } else {
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    }
                })
                .collect();
            tri2node2rgb
                .chunks_mut(9)
                .enumerate()
                .for_each(|(i_tri, v)| {
                    if let Some(cur_dist) = self.cur_dist {
                        if self.tri2dist[i_tri] <= cur_dist {
                            v.copy_from_slice(&[1.0, 0.7, 1.0, 1.0, 0.7, 1.0, 1.0, 0.7, 1.0]);
                        }
                    }
                });
            let gl = self.gl.clone().unwrap();
            self.drawer_tri
                .lock()
                .update_tri2node2rgb(&gl, &tri2node2rgb);
            self.is_updated_tri2flag = false;
        }
        let mat_modelview = self.trackball.mat4_col_major();
        let mat_projection = self.mat_projection;
        let z_flip = del_geo_core::mat4_col_major::from_diagonal(1., 1., -1., 1.);
        let mat_projection_for_opengl =
            del_geo_core::mat4_col_major::mult_mat_col_major(&z_flip, &mat_projection);
        let drawer_edge = self.drawer_edge.clone();
        let drawer_tri = self.drawer_tri.clone();
        let callback = egui::PaintCallback {
            rect,
            callback: std::sync::Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                unsafe {
                    gl.clear(glow::DEPTH_BUFFER_BIT);
                    gl.enable(glow::DEPTH_TEST);
                }
                drawer_edge
                    .lock()
                    .draw(painter.gl(), &mat_modelview, &mat_projection_for_opengl);
                drawer_tri
                    .lock()
                    .draw(painter.gl(), &mat_modelview, &mat_projection_for_opengl);
            })),
        };
        ui.painter().add(callback);
    }
}
