import cv2.cv2 as cv2
import numpy as np

from face_auth.config import Renderer as Config
from face_auth.model.geometry import Landmarks, Point, Rect


def draw_rect(frame: np.array, rect: Rect) -> None:
    cv2.rectangle(frame, rect.top_left, rect.bottom_right,
                  Config.Rect.COLOR, Config.Rect.THICKNESS, Config.Rect.LINE_TYPE)


def draw_landmarks(frame: np.array, landmarks: Landmarks, draw_bg=False, color=None) -> None:
    lm = landmarks
    alpha = Config.Landmarks.ALPHA

    if alpha <= 0.0:
        return

    if color is None:
        color = Config.Landmarks.BASE_COLOR
        mouth_color = Config.Landmarks.MOUTH_COLOR
        eye_color = Config.Landmarks.EYE_COLOR
    else:
        mouth_color = color
        eye_color = color

    if draw_bg:
        bg_alpha = 0.2
        overlay = frame.copy()
        cv2.fillPoly(overlay, Point.to_numpy(lm.outer_shape), color)
        cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

    thickness = Config.Landmarks.THICKNESS
    line_type = Config.Landmarks.LINE_TYPE

    overlay = frame.copy() if alpha < 1.0 else frame

    cv2.polylines(overlay, Point.to_numpy(lm.chin), False, color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.left_eyebrow), False, color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.right_eyebrow), False, color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.left_eye), True, eye_color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.right_eye), True, eye_color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.nose_bridge), False, color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.nose_tip), False, color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.top_lip), True, mouth_color, thickness, line_type)
    cv2.polylines(overlay, Point.to_numpy(lm.bottom_lip), True, mouth_color, thickness, line_type)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
