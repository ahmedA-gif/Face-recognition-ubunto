"""Direction semantics must follow the configured signed-distance orientation."""
from src.events.entry_exit_v2 import EntryExitEngineV2, TrackState


def _engine(direction):
    engine = EntryExitEngineV2(
        line_norm={"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5},
        entry_direction=direction,
    )
    engine._line = ((0.0, 50.0), (100.0, 50.0))
    return engine


def test_b_to_a_is_negative_to_positive_signed_distance():
    engine = _engine("B_to_A")
    assert engine._get_spatial_zone(-20) is TrackState.OUTSIDE
    assert engine._get_spatial_zone(20) is TrackState.INSIDE
    assert engine._normal_velocity((0, 10)) > 0


def test_a_to_b_reverses_inside_without_reversing_line_geometry():
    engine = _engine("A_to_B")
    assert engine._get_spatial_zone(20) is TrackState.OUTSIDE
    assert engine._get_spatial_zone(-20) is TrackState.INSIDE
    assert engine._normal_velocity((0, -10)) > 0
