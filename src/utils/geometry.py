from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Side = Literal["A", "B"]
Line = Tuple[Point, Point]
Polygon = Sequence[Sequence[float]]

# Region labels used by the Door Intelligence Engine.
REGION_OUTSIDE = "OUTSIDE"
REGION_DOOR = "DOOR"
REGION_INSIDE = "INSIDE"
# Priority order when a point falls inside more than one polygon: the door
# corridor (decision band) always wins, then inside, then the outside area.
REGION_PRIORITY = (REGION_DOOR, REGION_INSIDE, REGION_OUTSIDE)


def _cross(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Signed cross product (B-A) x (C-A)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def which_side(
    point: Point,
    line: Line,
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
    """Return transition string like 'A_to_B', or None if no side change."""
    if prev is None or prev == curr:
        return None
    return f"{prev}_to_{curr}"


def signed_distance(point: Point, line: Line) -> float:
    """Perpendicular signed distance from point to the infinite line.

    Positive => side A (left of directed segment), negative => side B.
    """
    (x1, y1), (x2, y2) = line
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1e-9
    return _cross(x1, y1, x2, y2, px, py) / length


def project_param(point: Point, line: Line) -> float:
    """Parametric position of the orthogonal projection of ``point`` on the line.

    t = 0 at the first endpoint, t = 1 at the second. Values outside [0, 1]
    mean the projection falls outside the door segment.
    """
    (x1, y1), (x2, y2) = line
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / denom


def is_near_segment(point: Point, line: Line, pad: float = 0.12) -> bool:
    """True when the point's projection lands on the segment (with end padding).

    ``pad`` is a fraction of segment length allowed past each endpoint so a
    person slightly outside the painted door edges still counts.
    """
    t = project_param(point, line)
    return -pad <= t <= 1.0 + pad


def foot_point(xyxy: Tuple[float, float, float, float] | list) -> Point:
    """Bottom-center of a person bbox — the feet, used for door crossings."""
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return ((x1 + x2) * 0.5, y2)


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True when segments ab and cd properly cross (endpoints excluded)."""
    def _cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1 = _cross(c[0], c[1], d[0], d[1], a[0], a[1])
    d2 = _cross(c[0], c[1], d[0], d[1], b[0], b[1])
    d3 = _cross(a[0], a[1], b[0], b[1], c[0], c[1])
    d4 = _cross(a[0], a[1], b[0], b[1], d[0], d[1])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def polygon_is_simple(polygon: Polygon) -> bool:
    """True when the polygon has no self-intersecting edges (O(n²) check)."""
    pts = list(polygon)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            if _segments_intersect(a, b, pts[j], pts[(j + 1) % n]):
                return False
    return True


def clean_polygon(polygon: Polygon, max_points: int = 12) -> Polygon:
    """Deduplicate + simplify a hand-drawn polygon into a valid, simple one.

    Calibration clicks produce jittery, self-intersecting polygons with tens
    of nearly-duplicate vertices; ``cv2.pointPolygonTest`` is undefined for
    those. This function:

    1. drops consecutive and exact duplicate points,
    2. Douglas-Peucker-simplifies down to ``max_points`` vertices,
    3. falls back to the convex hull if the result is still self-intersecting
       (hull is always simple) and drops polygons that collapse to < 3 points.

    Operates in the input coordinate space (normalized or pixel).
    """
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return []

    # 1. dedupe consecutive duplicates (keep first occurrence)
    keep = [pts[0]]
    for p in pts[1:]:
        if not np.allclose(p, keep[-1], atol=1e-6):
            keep.append(p)
    if len(keep) > 1 and np.allclose(keep[0], keep[-1], atol=1e-6):
        keep.pop()
    if len(keep) < 3:
        return []
    pts = np.asarray(keep)

    # 2. Douglas-Peucker simplification (approxPolyDP needs float32 contour).
    #    Gently increase epsilon until the result is simple and non-degenerate.
    #    epsilon as fraction of perimeter keeps it scale-invariant.
    perimeter = float(cv2.arcLength(pts.astype(np.float32).reshape(-1, 1, 2), True))
    simple = pts
    if perimeter > 0:
        for ep_f in (0.004, 0.008, 0.015, 0.025, 0.04):
            contour = cv2.approxPolyDP(
                pts.astype(np.float32).reshape(-1, 1, 2), perimeter * ep_f, True
            )
            cand = contour.reshape(-1, 2)
            if 3 <= len(cand) <= max_points and polygon_is_simple(cand):
                simple = cand
                break

    # 3. convex hull fallback for self-intersections / collapsed shapes — the
    #    hull is guaranteed simple and preserves the drawn region's extent.
    if len(simple) < 3 or not polygon_is_simple(simple):
        hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
        simple = hull.reshape(-1, 2)

    out = [[round(float(x), 4), round(float(y), 4)] for x, y in simple]
    return out if len(out) >= 3 else []


def point_in_polygon(point: Point, polygon: Polygon, on_edge_counts: bool = True) -> bool:
    """Test whether ``point`` lies inside ``polygon`` via cv2.pointPolygonTest.

    ``polygon`` is a sequence of (x, y) vertices in either pixel or normalized
    coordinates (the test is purely geometric). Points exactly on an edge are
    counted as inside when ``on_edge_counts`` is True.
    """
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    if on_edge_counts:
        return result >= 0
    return result > 0


def region_for_point(
    point: Point,
    regions: dict[str, Polygon],
    priority: Iterable[str] = REGION_PRIORITY,
) -> Optional[str]:
    """Label ``point`` with the first region (by ``priority``) containing it.

    Returns ``None`` if the point is outside every known region.
    """
    for name in priority:
        polygon = regions.get(name)
        if polygon and point_in_polygon(point, polygon):
            return name
    return None


def polygon_centroid(polygon: Polygon) -> Point:
    """Area-weighted centroid of a simple polygon (works for any vertex order)."""
    pts = np.asarray(polygon, dtype=np.float64)
    pts = np.vstack([pts, pts[0]])
    x, y = pts[:, 0], pts[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * cross.sum()
    if abs(area) < 1e-9:  # degenerate — fall back to vertex average
        return float(pts[:-1, 0].mean()), float(pts[:-1, 1].mean())
    cx = (x[:-1] + x[1:]).dot(cross) / (6.0 * area)
    cy = (y[:-1] + y[1:]).dot(cross) / (6.0 * area)
    return float(cx), float(cy)


def inward_normal(regions: dict[str, Polygon]) -> Optional[Tuple[float, float]]:
    """Unit vector pointing from the OUTSIDE region toward the INSIDE region.

    Used as the door's inward direction for the motion-dot-product gate.
    Returns ``None`` if either region is missing or the centroids coincide.
    """
    outside = regions.get(REGION_OUTSIDE)
    inside = regions.get(REGION_INSIDE)
    if not outside or not inside:
        return None
    ox, oy = polygon_centroid(outside)
    ix, iy = polygon_centroid(inside)
    dx, dy = ix - ox, iy - oy
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-9:
        return None
    return dx / norm, dy / norm
