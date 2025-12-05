use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::window::Window;

pub struct Content {
    pub tri2vtx: Vec<usize>,
    pub vtx2xyz: Vec<f32>,
    pub vtx2uv: Vec<f32>,
    pub bvhnodes: Vec<usize>,
    pub aabbs: Vec<f32>,
    pub tex_shape: (usize, usize),
    pub tex_data: Vec<f32>,
}

impl Content {
    fn new() -> Self {
        let (tri2vtx, vtx2xyz, vtx2uv) = {
            let mut obj = del_msh_cpu::io_wavefront_obj::WavefrontObj::<usize, f32>::new();
            obj.load("asset/spot/spot_triangulated.obj").unwrap();
            obj.unified_xyz_uv_as_trimesh()
        };
        let bvhnodes = del_msh_cpu::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let aabbs = del_msh_cpu::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        //println!("{:?}",img.color());
        let (tex_data, tex_shape, _bitdepth) =
            del_canvas::load_image_as_float_array("asset/spot/spot_texture.png").unwrap();
        //
        Self {
            tri2vtx,
            vtx2xyz,
            vtx2uv,
            bvhnodes,
            aabbs,
            tex_data,
            tex_shape,
        }
    }
}

impl del_gl_winit_glutin::viewer3d_for_image_generator::ImageGeneratorFrom3dCamPose for Content {
    fn compute_image(
        &mut self,
        img_shape: (usize, usize),
        cam_projection: &[f32; 16],
        cam_model: &[f32; 16],
    ) -> Vec<u8> {
        let transform_world2ndc =
            del_geo_core::mat4_col_major::mult_mat_col_major(cam_projection, cam_model);
        let transform_ndc2world =
            del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
        let mut pix2tri = vec![0usize; img_shape.0 * img_shape.1];
        del_msh_cpu::trimesh3_raycast::update_pix2tri(
            &mut pix2tri,
            &self.tri2vtx,
            &self.vtx2xyz,
            &self.bvhnodes,
            &self.aabbs,
            img_shape,
            &transform_ndc2world,
        );
        let img_data = del_msh_cpu::trimesh3_raycast::render_texture_from_pix2tri(
            img_shape,
            &transform_ndc2world,
            &self.tri2vtx,
            &self.vtx2xyz,
            &self.vtx2uv,
            &pix2tri,
            self.tex_shape,
            &self.tex_data,
            &del_msh_cpu::grid2::Interpolation::Bilinear,
        );
        let img_data: Vec<u8> = img_data
            .iter()
            .map(|v| (v * 255.0).clamp(0., 255.) as u8)
            .collect();
        img_data
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let template = glutin::config::ConfigTemplateBuilder::new()
        .with_alpha_size(8)
        .with_transparency(cfg!(cgl_backend));
    let display_builder = {
        let window_attributes = Window::default_attributes()
            .with_transparent(false)
            .with_title("01_texture_fullscrn")
            .with_inner_size(PhysicalSize {
                width: 600,
                height: 600,
            });
        glutin_winit::DisplayBuilder::new().with_window_attributes(Some(window_attributes))
    };
    let mut app = del_gl_winit_glutin::viewer3d_for_image_generator::Viewer3d::new(
        template,
        display_builder,
        Box::new(Content::new()),
    );
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app)?;
    app.appi.exit_state
}
