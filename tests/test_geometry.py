from src.utils.geometry import line_crossed, which_side


def test_which_side_and_cross():
    line = ((100.0, 0.0), (100.0, 200.0))
    assert which_side((50.0, 100.0), line) == "A"
    assert which_side((150.0, 100.0), line) == "B"
    assert line_crossed("A", "B") == "A_to_B"
    assert line_crossed("B", "A") == "B_to_A"
    assert line_crossed("A", "A") is None
