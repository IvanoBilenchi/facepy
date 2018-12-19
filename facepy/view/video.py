import cv2.cv2 as cv2
import numpy as np
from enum import Enum

import facepy.config as config
from . import color
from .fps_renderer import FPSRenderer
from .label_renderer import LabelPosition, LabelRenderer
from facepy.model import img
from facepy.model.fps_estimator import FPSEstimator
from facepy.model.geometry import Size


class VideoView:
    """Models the video view."""

    class Key(Enum):
        UNKNOWN = -2
        NONE = -1
        ENTER = 13
        ESC = 27
        SPACE = 32

        @staticmethod
        def from_int(value: int) -> 'VideoView.Key':
            try:
                ret_val = VideoView.Key(value)
            except ValueError:
                ret_val = VideoView.Key.UNKNOWN
            return ret_val

    # Public

    @property
    def top_text(self) -> str:
        return self.__top_renderer.text

    @top_text.setter
    def top_text(self, text: str) -> None:
        self.__top_renderer.text = text

    @property
    def bottom_text(self) -> str:
        return self.__bottom_renderer.text

    @bottom_text.setter
    def bottom_text(self, text: str) -> None:
        self.__bottom_renderer.text = text

    def __init__(self, show_fps: bool = True) -> None:
        self.__fps_estimator: FPSEstimator = None
        self.__fps_renderer: FPSRenderer = None
        self.__bottom_renderer = LabelRenderer(LabelPosition.BOTTOM_LEFT)
        self.__top_renderer = LabelRenderer(LabelPosition.TOP_LEFT)

        if show_fps:
            self.__fps_estimator = FPSEstimator()
            self.__fps_renderer = FPSRenderer()

        self.top_text = ''
        self.bottom_text = 'Ready'

    def temp_message(self, text: str, seconds: int) -> None:
        self.__bottom_renderer.temp_text(text, seconds)

    def display(self) -> None:
        cv2.namedWindow(config.Renderer.WINDOW_NAME)
        cv2.moveWindow(config.Renderer.WINDOW_NAME, 200, 150)

    def close(self) -> None:
        cv2.destroyWindow(config.Renderer.WINDOW_NAME)

    def get_key(self) -> Key:
        int_key = cv2.waitKey(1)
        key = VideoView.Key.from_int(int_key)

        if config.DEBUG and key != VideoView.Key.NONE:
            print('Pressed key: {}'.format(key if key != VideoView.Key.UNKNOWN else int_key))

        return key

    def render(self, frame: np.array) -> None:
        frame = img.resized(frame, config.Renderer.VIDEO_SIZE)
        self.__render_bars(frame)
        self.__render_labels(frame)
        cv2.imshow(config.Renderer.WINDOW_NAME, frame)

    # Private

    def __render_labels(self, frame: np.array) -> None:
        self.__bottom_renderer.render(frame)
        self.__top_renderer.render(frame)

        if self.__fps_renderer is not None:
            self.__fps_renderer.render(frame, self.__fps_estimator.tick_and_compute_fps())

    def __render_bars(self, frame: np.array) -> None:
        alpha = config.Renderer.Label.BAR_ALPHA

        if alpha <= 0.0:
            return

        bar_height = 2 * config.Renderer.Label.PADDING + config.Renderer.Label.FONT_HEIGHT
        size = Size.of_image(frame)

        overlay = frame.copy() if alpha < 1.0 else frame

        cv2.rectangle(overlay, (0, 0), (size.width, bar_height), color.BLACK, cv2.FILLED)
        cv2.rectangle(overlay, (0, size.height - bar_height), size, color.BLACK, cv2.FILLED)

        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
