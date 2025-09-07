import math
import numpy

import pyrr
from pyrr import Quaternion, Vector3, Matrix44


class ViewNavigation3:
    def __init__(self, view_height=1.0):
        self.view_height = view_height
        self.scale = 1.
        self.depth_ratio = 10.
        #
        self.translation = Vector3((0., 0., 0.), dtype=numpy.float64)
        self.quat = Quaternion((0., 0., 0., 1.), dtype=numpy.float64)
        #
        self.btn_left = False
        self.cursor_x = 0.
        self.cursor_y = 0.
        self.cursor_dx = 0.
        self.cursor_dy = 0.
        self.win_height = 480
        self.win_width = 640

    def update_cursor_position(self, x, y):
        fw = self.win_width
        fh = self.win_height
        x0 = self.cursor_x
        y0 = self.cursor_y
        self.cursor_x = (2.0 * x - fw) / fw
        self.cursor_y = (fh - 2.0 * y) / fh
        self.cursor_dx = self.cursor_x - x0
        self.cursor_dy = self.cursor_y - y0

    def projection_matrix(self) -> pyrr.Matrix44:
        asp = float(self.win_width) / float(self.win_height)
        mp = Matrix44(
            (1. / (self.view_height * asp), 0., 0., 0.,
             0., 1. / self.view_height, 0., 0.,
             0., 0., 1. / (self.view_height * self.depth_ratio), 0.,
             0., 0., 0., 1.))
        ms = Matrix44.from_scale((self.scale, self.scale, self.scale))
        return ms * mp

    def modelview_matrix(self) -> pyrr.Matrix44:
        mt = Matrix44.from_translation(self.translation)
        mr = Matrix44.from_quaternion(self.quat)
        return mt * mr

    def camera_rotation(self):
        dx = self.cursor_dx
        dy = self.cursor_dy
        a = math.sqrt(dx * dx + dy * dy)
        if a == 0.0:
            return
        dq = Quaternion.from_axis((dy, -dx, 0.))
        dq.normalize()
        self.quat = self.quat * dq

    def camera_translation(self):
        mp = self.projection_matrix()
        sx = (mp[3, 3] - mp[0, 3]) / mp[0, 0]
        sy = (mp[3, 3] - mp[1, 3]) / mp[1, 1]
        self.translation[0] += sx * self.cursor_dx
        self.translation[1] += sy * self.cursor_dy

    def picking_ray(self) -> (pyrr.Vector4, pyrr.Vector4):
        mmv = self.modelview_matrix()
        mp = self.projection_matrix()
        mmvpi = (mp * mmv).inverse
        q0 = mmvpi * pyrr.Vector4([self.cursor_x, self.cursor_y, +1., 1.])
        q1 = mmvpi * pyrr.Vector4([self.cursor_x, self.cursor_y, -1., 1.])
        return q0, q1 - q0
