import cv2.cv2 as cv2
import numpy as np
from time import perf_counter_ns

from face_auth.config import Renderer
from face_auth.model import img

Config = Renderer.FPS


class FPSRenderer:
    """Renders FPS stats."""

    # Public methods

    def __init__(self) -> None:
        self.fps = 0.0
        self.__frame_count = 0
        self.__frame_timestamp = perf_counter_ns()

        self.__label_width = cv2.getTextSize('FPS: 999.99', Config.FONT, Config.FONT_SCALE,
                                             Config.FONT_THICKNESS)[0][0]

    def render(self, frame: np.array) -> None:
        self.__frame_tick()

        fps = 'FPS: {0:.2f}'.format(self.fps)
        w, h = img.size(frame)
        cv2.putText(frame, fps, (w - self.__label_width, Config.FONT_HEIGHT + 10),
                    Config.FONT, Config.FONT_SCALE, Config.FONT_COLOR,
                    Config.FONT_THICKNESS, Config.LINE_TYPE)

    # Private methods

    def __frame_tick(self) -> None:
        self.__frame_count += 1

        current_ns = perf_counter_ns()
        delta = current_ns - self.__frame_timestamp

        if delta >= 500000000:
            self.fps = self.__frame_count / delta * 1000000000
            self.__frame_count = 0
            self.__frame_timestamp = current_ns
