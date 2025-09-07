"""
OpenGL drawer of uniform simplex mesh (triangles and line segments) with vertex color
"""

from pyrr import Matrix44
import numpy
import moderngl

class Drawer:

    def __init__(self, V:bytes, C:bytes, F:bytes):
        self.V = V
        self.C = C
        self.F = F

    def init_gl(self, ctx):
        self.prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_color;
                out vec3 v_vert;
                out vec3 v_color;
                void main() {
                    v_vert = in_position;
                    v_color = in_color;
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_vert;
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            '''
        )
        self.mvp = self.prog['Mvp']

        vao_content = [
            (ctx.buffer(self.V), '3f', 'in_position'),
            (ctx.buffer(self.C), '3f', 'in_color')
        ]
        index_buffer = ctx.buffer(self.F)

        self.vao = ctx.vertex_array(
            self.prog, vao_content, index_buffer, 4)

    def paint_gl(self, mvp: Matrix44):
        self.mvp.value = tuple(mvp.flatten())
        self.vao.render(mode=moderngl.TRIANGLES)