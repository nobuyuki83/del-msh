from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rect:
    w: int
    h: int


@dataclass(frozen=True)
class Placement:
    index: int
    x: int
    y: int
    w: int
    h: int


def overlaps(a: Placement, b: Placement) -> bool:
    return not (
        a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
    )


def pack_rectangles(
    rectangles: list[Rect],
    bin_width: int,
    bin_height: int,
) -> Optional[list[Placement]]:
    """
    指定された bin_width × bin_height に全矩形が入るか調べる。
    回転は許さない。
    """

    # 大きい矩形から置くと枝刈りしやすい
    order = sorted(
        range(len(rectangles)),
        key=lambda i: (
            rectangles[i].w * rectangles[i].h,
            max(rectangles[i].w, rectangles[i].h),
        ),
        reverse=True,
    )

    placed: list[Placement] = []

    def search(depth: int) -> bool:
        if depth == len(order):
            return True

        index = order[depth]
        rect = rectangles[index]

        candidate_x = {0}
        candidate_y = {0}

        for p in placed:
            candidate_x.add(p.x + p.w)
            candidate_y.add(p.y + p.h)

        # 左下側から試す
        positions = sorted(
            ((x, y) for y in candidate_y for x in candidate_x),
            key=lambda pos: (pos[1], pos[0]),
        )

        for x, y in positions:
            if x + rect.w > bin_width:
                continue

            if y + rect.h > bin_height:
                continue

            candidate = Placement(
                index=index,
                x=x,
                y=y,
                w=rect.w,
                h=rect.h,
            )

            if any(overlaps(candidate, p) for p in placed):
                continue

            placed.append(candidate)

            if search(depth + 1):
                return True

            placed.pop()

        return False

    if not search(0):
        return None

    return sorted(placed, key=lambda p: p.index)


def find_minimum_container(
    rectangles: list[Rect],
) -> tuple[int, int, list[Placement]]:
    """
    外形面積 W * H を最小化する。
    面積が同じなら W + H が小さいものを選ぶ。
    """

    if not rectangles:
        return 0, 0, []

    total_area = sum(r.w * r.h for r in rectangles)

    # 回転なしなので、少なくとも最大の矩形幅が必要
    min_width = max(r.w for r in rectangles)

    # 全矩形を横一列に並べた幅
    max_width = sum(r.w for r in rectangles)

    # 全矩形を縦一列に並べた配置を初期解とする
    best_width = max(r.w for r in rectangles)
    best_height = sum(r.h for r in rectangles)

    best_placement = pack_rectangles(
        rectangles,
        best_width,
        best_height,
    )

    assert best_placement is not None

    best_area = best_width * best_height

    for width in range(min_width, max_width + 1):
        # 面積下限から高さの下限を求める
        min_height = max(
            max(r.h for r in rectangles),
            (total_area + width - 1) // width,
        )

        # 現在の最良面積を超えない高さだけ調べればよい
        max_height = best_area // width

        if min_height > max_height:
            continue

        # 高さを小さい方から試す。
        # 最初に入った高さが、この width に対する最小高さ。
        for height in range(min_height, max_height + 1):
            placement = pack_rectangles(
                rectangles,
                width,
                height,
            )

            if placement is None:
                continue

            area = width * height

            candidate_key = (
                area,
                width + height,
                abs(width - height),
            )

            best_key = (
                best_area,
                best_width + best_height,
                abs(best_width - best_height),
            )

            if candidate_key < best_key:
                best_width = width
                best_height = height
                best_area = area
                best_placement = placement

            # この幅については、これ以上高さを増やす必要はない
            break

    return best_width, best_height, best_placement


if __name__ == "__main__":
    rectangles = [
        Rect(4, 3),
        Rect(3, 2),
        Rect(2, 2),
        Rect(1, 4),
    ]

    width, height, placements = find_minimum_container(rectangles)

    print(f"container: {width} x {height}")
    print(f"area: {width * height}")

    for p in placements:
        print(f"rect {p.index}: position=({p.x}, {p.y}), size=({p.w}, {p.h})")
