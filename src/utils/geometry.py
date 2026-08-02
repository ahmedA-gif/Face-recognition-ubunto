from __future__ import annotations

from typing import Literal, Tuple

Point = Tuple[float, float]
Side = Literal["A", "B"]


def _cross(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Signed cross product (B-A) x (C-A)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def which_side(
    point: Point,
    line: tuple[Point, Point],
) -> Side:
    """
    Side A = left of directed line (x1,y1)->(x2,y2)
    Side B = right
    """
    (x1, y1), (x2, y2) = line
    px, py = point
    c = _cross(x1, y1, x2, y2, px, py)
    return "A" if c >= 0 else "B"


def line_crossed(prev: Side | None, curr: Side) -> str | None:
    """Return 'entry' or 'exit' mapping is applied by caller via settings."""
    if prev is None or prev == curr:
        return None
    return f"{prev}_to_{curr}"
