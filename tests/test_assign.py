import numpy as np

from src.utils.assign import face_to_head_distance


def test_face_is_assigned_by_head_proximity_not_person_box_area():
    person = np.array([0, 0, 100, 200], dtype=np.float32)
    face_near_head = np.array([40, 25, 60, 45], dtype=np.float32)
    face_near_feet = np.array([40, 155, 60, 175], dtype=np.float32)
    assert face_to_head_distance(face_near_head, person) < face_to_head_distance(face_near_feet, person)
