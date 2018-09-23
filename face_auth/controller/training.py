import cv2.cv2 as cv2
import numpy as np

from face_auth.model import detector
from face_auth.model.detector import DetectionAlgo


# Public functions


def process_frame(frame: np.array, key_press: int) -> None:
    del key_press  # Unused
    __highlight_faces(frame)


# Private functions


def __highlight_faces(frame: np.array) -> None:
    pass
    rect_color = (0, 0, 255)
    line_width = 3

    for face in detector.detect_faces(frame, DetectionAlgo.HOG):
        cv2.rectangle(frame, face.top_left, face.bottom_right, rect_color, line_width)
