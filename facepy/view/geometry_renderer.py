import cv2.cv2 as cv2
import numpy as np
from typing import Iterable

from facepy.config import Renderer as Config
from facepy.model.geometry import Landmarks, Point


def draw_points(image: np.array, points: Iterable[Point]) -> None:
    """Draws points on the specified image."""
    thickness = Config.Landmarks.THICKNESS
    color = Config.Landmarks.BASE_COLOR
    for pt in points:
        cv2.circle(image, pt, thickness, color, thickness)


def draw_landmarks(image: np.array, landmarks: Landmarks,
                   points=False, draw_bg=False, color=None) -> None:
    """Draws facial landmarks on the specified image."""
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
        overlay = image.copy()
        cv2.fillPoly(overlay, Point.to_numpy(lm.outer_shape), color)
        cv2.addWeighted(overlay, bg_alpha, image, 1.0 - bg_alpha, 0, image)

    thickness = Config.Landmarks.THICKNESS
    line_type = Config.Landmarks.LINE_TYPE

    overlay = image.copy() if alpha < 1.0 else image

    for pts in [lm.chin, lm.left_eyebrow, lm.right_eyebrow, lm.nose_bridge, lm.nose_tip]:
        if points:
            for point in pts:
                cv2.circle(overlay, point, thickness, color, thickness, line_type)
        else:
            cv2.polylines(overlay, Point.to_numpy(pts), False, color, thickness, line_type)

    for pts in [lm.left_eye, lm.right_eye]:
        if points:
            for point in pts:
                cv2.circle(overlay, point, thickness, eye_color, thickness, line_type)
        else:
            cv2.polylines(overlay, Point.to_numpy(pts), True, eye_color, thickness, line_type)

    for pts in [lm.top_lip, lm.bottom_lip]:
        if points:
            for point in pts:
                cv2.circle(overlay, point, thickness, mouth_color, thickness, line_type)
        else:
            cv2.polylines(overlay, Point.to_numpy(pts), True, mouth_color, thickness, line_type)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
