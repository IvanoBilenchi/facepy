import cv2.cv2 as cv2
import numpy as np
from time import perf_counter_ns

from face_auth.model import img


class FPSRenderer:
    """Renders FPS stats."""

    # Public constants

    FONT_HEIGHT = 14
    FONT_COLOR = (0, 255, 0)
    FONT_SCALE = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, FONT_HEIGHT)

    # Public methods

    def __init__(self) -> None:
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
        fps = 'FPS: {0:.2f}'.format(self.fps)
        w, h = img.size(frame)
        cv2.putText(frame, fps, (w - 140, self.FONT_HEIGHT + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_COLOR)
