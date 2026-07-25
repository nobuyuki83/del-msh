import math

import torch


def _normalized(v: torch.Tensor) -> torch.Tensor:
    v = v.view(-1, 3)
    return v / torch.linalg.vector_norm(v, dim=-1)[..., None]


def perspective(
    fovy: torch.Tensor,
    aspect: torch.Tensor,
    near=torch.tensor([1e-2], dtype=torch.float32),
    far=torch.tensor([1e4], dtype=torch.float32),
) -> torch.Tensor:
    """Create perspective projection matrix.

    Arguments:
        fovy: vertical field-of-view values with shape (B,).
        aspect: x/y aspect ratios with shape (B,).
        near, far: near/far clipping distances with shape (B,).

    Returns:
        homography matrices with shape (B, 4, 4).
    """

    B = fovy.shape[0]
    device = fovy.device

    assert fovy.dtype == torch.float32
    assert aspect.device == device and aspect.dtype == torch.float32
    assert near.device == device and near.dtype == torch.float32
    assert far.device == device and far.dtype == torch.float32

    tan_theta = torch.tan(0.5 * math.pi * fovy / 180.0)
    right = aspect * near * tan_theta
    top = near * tan_theta

    m = torch.zeros(B, 4, 4, dtype=torch.float32, device=device)
    m[..., 0, 0] = near / right
    m[..., 1, 1] = near / top
    m[..., 2, 2] = -(far + near) / (far - near)
    m[..., 2, 3] = -2.0 * far * near / (far - near)
    m[..., 3, 2] = -1.0

    return m


def look_at(
    origin: torch.Tensor, target: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """Create look-at transformation matrix.

    Arguments:
        origin: camera origins with shape (B, 3).
        target: camera targets with shape (B, 3).
        up: camera up directions with shape (B, 3).

    Returns:
        homography matrices with shape (B, 4, 4).
    """

    B = origin.shape[0]
    device = origin.device

    assert origin.dtype == torch.float32
    assert target.device == device and target.dtype == torch.float32
    assert up.device == device and up.dtype == torch.float32

    # right-handed coordinates
    ez = _normalized(origin - target)
    ex = _normalized(torch.linalg.cross(up, ez))
    ey = _normalized(torch.linalg.cross(ez, ex))

    m = torch.zeros(B, 4, 4, dtype=torch.float32, device=device)
    m[..., 0, 0], m[..., 0, 1], m[..., 0, 2], m[..., 0, 3] = (
        ex[..., 0],
        ex[..., 1],
        ex[..., 2],
        -torch.linalg.vecdot(origin, ex),
    )
    m[..., 1, 0], m[..., 1, 1], m[..., 1, 2], m[..., 1, 3] = (
        ey[..., 0],
        ey[..., 1],
        ey[..., 2],
        -torch.linalg.vecdot(origin, ey),
    )
    m[..., 2, 0], m[..., 2, 1], m[..., 2, 2], m[..., 2, 3] = (
        ez[..., 0],
        ez[..., 1],
        ez[..., 2],
        -torch.linalg.vecdot(origin, ez),
    )
    m[..., 3, 3] = 1.0

    return m


class PerspectiveCamera:
    def __init__(
        self,
        fovy: torch.Tensor = torch.tensor([40.0], dtype=torch.float32),
        aspect: torch.Tensor = torch.tensor([1.0], dtype=torch.float32),
        near: torch.Tensor = torch.tensor([1e-2], dtype=torch.float32),
        far: torch.Tensor = torch.tensor([100.0], dtype=torch.float32),
    ) -> None:
        """Create camera object

        Arguments:
            fovy: vertical field-of-view tensor (shape = [1,]).
            aspect: x/y aspect ratio tensor (shape = [1,]).
            near: near clipping distance tensor (shape = [1,]).
            far: far clipping distance tensor (shape = [1,]).
        """

        self.fovy = fovy.view(1).cpu()
        self.aspect = aspect.view(1).cpu()
        self.near = near.view(1).cpu()
        self.far = far.view(1).cpu()

    def rotate(self, origin: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Create camera matrix from rotation angles.

        Arguments:
            origin: camera origin tensor (shape = [num, 3]).
            angles: Euler angle tensor (shape = [num, 3]).

        Returns:
            homography matrix tensor (shape = [num, 4, 4]).
        """

        origin = origin.view(-1, 3).cpu()
        angles = angles.view(-1, 3).cpu()

        m = rotation(angles).squeeze(0)
        up = torch.matmul(m, torch.tensor([[0, 1, 0]], dtype=torch.float32))
        to = torch.matmul(m, torch.tensor([[0, 0, -1]], dtype=torch.float32))

        return self.look_at(origin, origin + to, up)

    def look_at(
        self,
        origin: torch.Tensor,
        target: torch.Tensor,
        up=torch.tensor([0, 1, 0], dtype=torch.float32),
    ) -> torch.Tensor:
        """Create camera matrix from look-at transformation.

        Arguments:
            origin: camera origin tensor (shape = [num, 3]).
            target: camera target tensor (shape = [num, 3]).
            up: camera up tensor (shape = [num, 3]).

        Returns:
            homography matrix tensor (shape = [num, 4, 4]).
        """

        origin = origin.view(-1, 3).cpu()
        target = target.view(-1, 3).cpu()
        up = _normalized(up.view(-1, 3)).cpu()

        proj = perspective(self.fovy, self.aspect, self.near, self.far)
        view = look_at(origin, target, up)

        return torch.matmul(proj, view)

    def ndc2depth(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute linear depth from perspective NDC position.

        Arguments:
            pos: vertex position tensor of size (*, N, 4).
        Returns:
            depth tensor of size (*, N).
        """

        tan_theta = torch.tan(0.5 * math.pi * self.fovy / 180.0)
        mx = self.aspect * tan_theta
        my = tan_theta

        device = pos.device
        x = mx.to(device) * pos[..., 0]
        y = my.to(device) * pos[..., 1]
        w = pos[..., 3]
        return torch.sqrt(x.square() + y.square() + w.square())
