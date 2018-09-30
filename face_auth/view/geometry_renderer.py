import cv2.cv2 as cv2
import numpy as np

from face_auth.config import Renderer as Config
from face_auth.model.geometry import Landmarks, Rect


def draw_rect(frame: np.array, rect: Rect) -> None:
    cv2.rectangle(frame, rect.top_left, rect.bottom_right,
                  Config.Rect.COLOR, Config.Rect.THICKNESS, Config.Rect.LINE_TYPE)


def draw_landmarks(frame: np.array, landmarks: Landmarks) -> None:
    lm = landmarks

    color = Config.Landmarks.BASE_COLOR
    mouth_color = Config.Landmarks.MOUTH_COLOR
    eye_color = Config.Landmarks.EYE_COLOR
    thickness = Config.Landmarks.THICKNESS
    line_type = Config.Landmarks.LINE_TYPE

    cv2.polylines(frame, Landmarks.to_numpy(lm.chin), False, color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.left_eyebrow), False, color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.right_eyebrow), False, color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.left_eye), True, eye_color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.right_eye), True, eye_color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.nose_bridge), False, color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.nose_tip), False, color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.top_lip), True, mouth_color, thickness, line_type)
    cv2.polylines(frame, Landmarks.to_numpy(lm.bottom_lip), True, mouth_color, thickness, line_type)
