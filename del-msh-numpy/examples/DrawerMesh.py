"""
OpenGL drawer of uniform simplex mesh (triangles and line segments)
"""

import typing
from pyrr import Matrix44
import numpy
import moderngl


class ElementInfo:

    def __init__(self, index: numpy.ndarray, mode, color: tuple):
        self.vao = None
        # index should be numpy.uint32
        if index.dtype == numpy.uint32:
            self.index = index
        else:
            self.index = index.astype(numpy.uint32)
        self.mode = mode
        self.color = color


class Drawer:

    def __init__(self, vtx2xyz: numpy.ndarray, list_elem2vtx: typing.List[ElementInfo]):
        assert len(vtx2xyz.shape) == 2
        if vtx2xyz.dtype == numpy.float32:
            self.vtx2xyz = vtx2xyz
        else:
            self.vtx2xyz = vtx2xyz.astype(numpy.float32)
        self.list_elem2vtx = list_elem2vtx
        self.vao_content = None

    def init_gl(self, ctx: moderngl.Context):
        self.prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 color;                
                out vec4 f_color;
                void main() {
                    f_color = vec4(color, 1.0);
                }
            '''
        )

        self.vao_content = [
            (ctx.buffer(self.vtx2xyz.tobytes()), f'{self.vtx2xyz.shape[1]}f', 'in_position'),
        ]
        #del self.vtx2xyz
        for el in self.list_elem2vtx:
            index_buffer = ctx.buffer(el.index.tobytes())
            el.vao = ctx.vertex_array(
                self.prog, self.vao_content, index_buffer, 4
            )
            #del el.index

    def update_position(self, V: numpy.ndarray):
        if V.dtype != numpy.float32:
            V1 = V.astype(numpy.float32)
        else:
            V1 = V
        if self.vao_content is not None:
            vbo = self.vao_content[0][0]
            vbo.write(V1.tobytes())

    def paint_gl(self, mvp: Matrix44):
        self.prog['Mvp'].value = tuple(mvp.flatten())
        for el in self.list_elem2vtx:
            self.prog['color'].value = el.color
            el.vao.render(mode=el.mode)
