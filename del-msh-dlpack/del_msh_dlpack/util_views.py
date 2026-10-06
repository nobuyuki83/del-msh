import math
import typing

#
import torch

#
from del_msh_dlpack.util_packing import Rect, find_minimum_container

"""正投影ビューを生成し、1 枚のアトラスに詰め込むための補助関数群。

主な処理の流れは次のとおり。
1. 入力されたバウンディングボックスから 6 方向の軸揃えビューを作る。
2. 各ビューについて world-to-NDC 行列と画像サイズを計算する。
3. 得られた複数の画像を 1 枚の出力テクスチャにパックする。
"""

# ---------------------------------------------------------------------------
# 同次座標系の 4x4 変換行列ヘルパー
# すべての関数はスカラー tensor (0-d) を受け取り、(4, 4) 行列を返す。
# 座標系の約束は列ベクトルで、p_ndc = M @ p_world とする。
# ---------------------------------------------------------------------------


def rot_x(rx: torch.Tensor) -> torch.Tensor:
    """X 軸まわりに rx ラジアン回転する 4x4 同次変換行列を返す。"""
    c = torch.cos(rx)
    s = torch.sin(rx)
    R = torch.eye(4, dtype=rx.dtype, device=rx.device)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def rot_y(ry: torch.Tensor) -> torch.Tensor:
    """Y 軸まわりに ry ラジアン回転する 4x4 同次変換行列を返す。"""
    c = torch.cos(ry)
    s = torch.sin(ry)
    R = torch.eye(4, dtype=ry.dtype, device=ry.device)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def rot_z(rz: torch.Tensor) -> torch.Tensor:
    """Z 軸まわりに rz ラジアン回転する 4x4 同次変換行列を返す。"""
    c = torch.cos(rz)
    s = torch.sin(rz)
    R = torch.eye(4, dtype=rz.dtype, device=rz.device)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def translation_matrix(t: torch.Tensor) -> torch.Tensor:
    """
    4x4 の同次平行移動行列を返す。


    Args:
        t: shape (3,) - 平行移動ベクトル [tx, ty, tz]
    """
    T = torch.eye(4, dtype=t.dtype, device=t.device)
    T[:3, 3] = t
    return T


def scale_matrix(s: torch.Tensor) -> torch.Tensor:
    """
    4x4 の非一様スケール行列を返す。


    Args:
        s: shape (3,) - スケール係数 [sx, sy, sz]
    """
    S = torch.eye(4, dtype=s.dtype, device=s.device)
    S[0, 0] = s[0]
    S[1, 1] = s[1]
    S[2, 2] = s[2]
    return S


def make_world2ndc_from_ortho_aabb(
    aabb: torch.Tensor,
    aabb_image: typing.Sequence[float],
    rot: torch.Tensor,
    fit_ratio: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """回転後の 1 視点に対する正投影の world-to-NDC 変換を作る。

    入力 AABB は物体座標系では軸揃えのままとし、``rot`` で視線方向を
    切り替える。計算されたスケールは、指定した画像アスペクト比の中へ
    投影結果が少し余白を持って収まるように調整される。
    """
    assert aabb.shape == (2, 3)
    assert len(aabb_image) == 3
    r_aabb_img = (
        (rot @ torch.Tensor([aabb_image[0], aabb_image[1], aabb_image[2], 1.0]))
        .abs()
        .round()
        .int()
        .tolist()
    )
    img_shape = (r_aabb_img[0], r_aabb_img[1])
    #
    center = aabb.mean(axis=0).cpu()
    T = translation_matrix(-center)  # 物体中心を原点へ移す
    aabb_lens = (aabb[1] - aabb[0]).tolist()
    r_aabb_lens = (
        rot @ torch.tensor([aabb_lens[0], aabb_lens[1], aabb_lens[2], 0.0])
    ).abs()
    scale_z = 2.0 * fit_ratio / r_aabb_lens[2]
    scale_x = img_shape[0] / r_aabb_lens[0]
    scale_y = img_shape[1] / r_aabb_lens[1]
    if scale_x > scale_y:
        scale = 2.0 / r_aabb_lens[1] * fit_ratio
        P = torch.tensor([scale * img_shape[1] / img_shape[0], scale, scale_z])
    else:
        scale = 2.0 / r_aabb_lens[0] * fit_ratio
        P = torch.tensor([scale, scale * img_shape[0] / img_shape[1], scale_z])
    P = scale_matrix(P)
    world2ndc = P @ rot @ T
    return world2ndc, img_shape


def make_world2ndc_imgsize(
    aabb: torch.Tensor,
    base_resolution: int,
    fit_ratio: float,
) -> list[tuple[torch.Tensor, tuple[int, int]]]:
    """バウンディングボックスに対する 6 方向の標準的な正投影ビューを作る。

    各ビューには変換行列と画像サイズの両方を持たせる。画像サイズは
    AABB の縦横比に応じて ``base_resolution`` の整数倍で決まる。
    """
    assert aabb.shape == (2, 3)
    aabb_lens = aabb[1] - aabb[0]
    assert aabb_lens[0] >= 0 and aabb_lens[1] >= 0 and aabb_lens[2] >= 0
    aabb_lens_min = aabb_lens.min()
    aabb_img = ((aabb_lens / aabb_lens_min).int() * base_resolution).tolist()

    list_view = []
    # 前、上、下、右、後、左の 6 視点を並べる。
    rot_none = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    list_view.append(
        make_world2ndc_from_ortho_aabb(aabb, aabb_img, rot_none, fit_ratio)
    )
    #
    rot_x90 = rot_x(torch.tensor(math.pi * 0.5))  # 上方向から見る
    list_view.append(make_world2ndc_from_ortho_aabb(aabb, aabb_img, rot_x90, fit_ratio))
    #
    rot_xm90 = rot_x(torch.tensor(-math.pi * 0.5))  # 下方向から見る
    list_view.append(
        make_world2ndc_from_ortho_aabb(aabb, aabb_img, rot_xm90, fit_ratio)
    )
    #
    rot_y90 = rot_y(torch.tensor(math.pi * 0.5))  # 右方向から見る
    list_view.append(make_world2ndc_from_ortho_aabb(aabb, aabb_img, rot_y90, fit_ratio))
    #
    rot_y180 = rot_y(torch.tensor(math.pi))  # 後方向から見る
    list_view.append(
        make_world2ndc_from_ortho_aabb(aabb, aabb_img, rot_y180, fit_ratio)
    )
    #
    rot_y270 = rot_y(torch.tensor(math.pi * 1.5))  # 左方向から見る
    list_view.append(
        make_world2ndc_from_ortho_aabb(aabb, aabb_img, rot_y270, fit_ratio)
    )

    return list_view


def packing_for_images(
    list_image_shape: typing.Sequence[tuple[int, int]],
    base_resolution: int,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """各ビューの画像矩形を base_resolution 単位で 1 枚のアトラスへ詰める。"""
    list_rect = []
    for img_shape in list_image_shape:
        w, h = img_shape
        assert w % base_resolution == 0
        assert h % base_resolution == 0
        iw = w // base_resolution
        ih = h // base_resolution
        list_rect.append(Rect(iw, ih))
    #
    w, h, placements = find_minimum_container(list_rect)
    list_viewport = []
    for i_placement, placement in enumerate(placements):
        assert placement.index == i_placement
        assert placement.w == list_image_shape[i_placement][0] // base_resolution
        assert placement.h == list_image_shape[i_placement][1] // base_resolution
        viewport = (
            placement.x * base_resolution,
            placement.y * base_resolution,
            list_image_shape[i_placement][0],
            list_image_shape[i_placement][1],
        )
        list_viewport.append(viewport)
    return (w * base_resolution, h * base_resolution), list_viewport


def make_views(
    aabb: torch.Tensor,
    base_resolution: int,
    fit_ratio: float,
) -> tuple[tuple[int, int], list[tuple[torch.Tensor, tuple[int, int, int, int]]]]:
    """パック後のアトラスサイズと各ビューの変換・ビューポートを返す。"""
    views = make_world2ndc_imgsize(aabb, base_resolution, fit_ratio)
    list_world2ndc, list_image_shape = zip(*views)
    image_shape, list_viewports = packing_for_images(list_image_shape, base_resolution)
    views = list(zip(list_world2ndc, list_viewports))
    return image_shape, views
