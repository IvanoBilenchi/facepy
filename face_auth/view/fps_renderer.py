import cv2.cv2 as cv2
import numpy as np
from time import perf_counter_ns

from face_auth.model import img


class FPSRenderer:
    """Renders FPS stats."""

    def __init__(self, cam: cv2.VideoCapture) -> None:
        self.max_fps = cam.get(cv2.CAP_PROP_FPS)
        self.fps = 0.0

        self.__frame_count = 0
        self.__frame_timestamp = perf_counter_ns()

    def frame_tick(self) -> None:
        self.__frame_count += 1

        current_ns = perf_counter_ns()
        delta = current_ns - self.__frame_timestamp

        if delta >= 500000000:
            self.fps = self.__frame_count / delta * 1000000000
            self.__frame_count = 0
            self.__frame_timestamp = current_ns

    def render(self, frame: np.array) -> None:
        font_color = (0, 255, 0)
        font_height = 14

        max_fps = 'MAX FPS: {0:.2f}'.format(self.max_fps)
        fps = 'FPS: {0:.2f}'.format(self.fps)

        w, h = img.size(frame)
        font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, font_height)

        cv2.putText(frame, max_fps, (w - 180, font_height + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color)
        cv2.putText(frame, fps, (w - 180, 2 * font_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color)
