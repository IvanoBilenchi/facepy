import cv2.cv2 as cv2
import numpy as np
from enum import Enum

import face_auth.config as config
from face_auth.model import img
from face_auth.model.fps_estimator import FPSEstimator
from .fps_renderer import FPSRenderer


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

    # Public methods

    def __init__(self, show_fps: bool = True) -> None:
        self.__fps_estimator: FPSEstimator = None
        self.__fps_renderer: FPSRenderer = None

        if show_fps:
            self.__fps_estimator = FPSEstimator()
            self.__fps_renderer = FPSRenderer()

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
        if self.__fps_renderer is not None:
            self.__fps_renderer.render(frame, self.__fps_estimator.tick_and_compute_fps())
        cv2.imshow(config.Renderer.WINDOW_NAME, frame)
