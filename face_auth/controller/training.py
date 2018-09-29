import cv2.cv2 as cv2
import numpy as np

from face_auth.model import detector
from face_auth.model.detector import DetectionAlgo
from face_auth.model.geometry import Landmarks, Rect


# Public functions


def process_frame(frame: np.array, key_press: int) -> None:
    del key_press  # Unused
    __highlight_faces(frame)


# Private functions


def __highlight_faces(frame: np.array) -> None:
    rect_color = (255, 0, 0)
    line_width = 3

    for face in detector.detect_faces(frame, DetectionAlgo.HOG):
        cv2.rectangle(frame, face.top_left, face.bottom_right, rect_color, line_width)
        __draw_landmarks(frame, face)


def __draw_landmarks(frame: np.array, rect: Rect) -> None:
    color = (0, 255, 0)
    mouth_color = (0, 0, 255)
    eye_color = (255, 255, 255)
    thickness = 2

    lm = detector.detect_landmarks(frame, rect)

    cv2.polylines(frame, Landmarks.to_numpy(lm.chin), False, color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.left_eyebrow), False, color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.right_eyebrow), False, color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.left_eye), True, eye_color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.right_eye), True, eye_color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.nose_bridge), False, color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.nose_tip), False, color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.top_lip), True, mouth_color, thickness)
    cv2.polylines(frame, Landmarks.to_numpy(lm.bottom_lip), True, mouth_color, thickness)
