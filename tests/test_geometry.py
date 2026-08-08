import pytest

from src.utils.geometry import (
    foot_point,
    inward_normal,
    is_near_segment,
    line_crossed,
    point_in_polygon,
    polygon_centroid,
    project_param,
    region_for_point,
    signed_distance,
    which_side,
)


# ── point-in-polygon / region classifier (Door Intelligence Engine) ───────────

def _zones():
    return {
        "OUTSIDE": [[15, 5], [85, 5], [85, 48], [15, 48]],
        "DOOR": [[18, 48], [82, 48], [82, 62], [18, 62]],
        "INSIDE": [[5, 62], [95, 62], [95, 98], [5, 98]],
    }


def test_point_in_polygon_inside_edge_outside():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 50), poly)
    assert point_in_polygon((0, 50), poly)          # on edge counts inside
    assert not point_in_polygon((150, 50), poly)


def test_region_for_point_priority():
    zones = _zones()
    # A point in the corridor band → DOOR wins even though INSIDE also covers y=60.
    assert region_for_point((50, 55), zones) == "DOOR"
    # Below the corridor → INSIDE.
    assert region_for_point((50, 80), zones) == "INSIDE"
    # Above the corridor → OUTSIDE.
    assert region_for_point((50, 20), zones) == "OUTSIDE"
    # Outside everything → None.
    assert region_for_point((150, 150), zones) is None


def test_polygon_centroid_and_inward_normal():
    zones = _zones()
    cx, cy = polygon_centroid([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert (cx, cy) == pytest.approx((50.0, 50.0))
    nx, ny = inward_normal(zones)
    # Inside centroid is below outside centroid → normal points +y (down-screen).
    assert ny > 0
    assert abs(nx) < 1e-9


def test_which_side_and_cross():
    line = ((100.0, 0.0), (100.0, 200.0))
    assert which_side((50.0, 100.0), line) == "A"
    assert which_side((150.0, 100.0), line) == "B"
    assert line_crossed("A", "B") == "A_to_B"
    assert line_crossed("B", "A") == "B_to_A"
    assert line_crossed("A", "A") is None


def test_signed_distance_horizontal_line():
    # Horizontal door at y=55 (camera inside, door at top of frame).
    line = ((20.0, 55.0), (80.0, 55.0))
    # Above the line (outside) → negative (zone B).
    assert signed_distance((50.0, 40.0), line) < 0
    # Below the line (inside) → positive (zone A).
    assert signed_distance((50.0, 70.0), line) > 0
    # On the line → zero.
    assert signed_distance((50.0, 55.0), line) == 0.0


def test_project_param_on_segment():
    line = ((20.0, 55.0), (80.0, 55.0))
    assert project_param((50.0, 55.0), line) == pytest.approx(0.5)
    assert project_param((20.0, 55.0), line) == pytest.approx(0.0)
    assert project_param((80.0, 55.0), line) == pytest.approx(1.0)
    # Off the ends of the segment.
    assert project_param((10.0, 55.0), line) < 0
    assert project_param((90.0, 55.0), line) > 1.0


def test_is_near_segment_gate():
    line = ((20.0, 55.0), (80.0, 55.0))
    assert is_near_segment((50.0, 55.0), line, pad=0.12)
    # Just outside the segment ends (beyond padding) → rejected.
    assert not is_near_segment((8.0, 55.0), line, pad=0.12)
    assert not is_near_segment((95.0, 55.0), line, pad=0.12)
    # Inside padding overshoot → still accepted.
    assert is_near_segment((14.0, 55.0), line, pad=0.12)


def test_foot_point_is_bottom_center():
    assert foot_point([10.0, 20.0, 30.0, 80.0]) == (20.0, 80.0)
    assert foot_point([0.0, 0.0, 10.0, 10.0]) == (5.0, 10.0)
