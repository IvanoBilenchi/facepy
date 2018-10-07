import cv2.cv2 as cv2
import numpy as np
from enum import Enum

from face_auth.config import Renderer as Config
from face_auth.model import img
from .fps_renderer import FPSRenderer


class VideoView:
    """Models the video view."""

    class Key(Enum):
        NONE = 0
        ESC = 27
        SPACE = 32

        @staticmethod
        def from_int(value: int) -> 'VideoView.Key':
            try:
                ret_val = VideoView.Key(value)
            except ValueError:
                ret_val = VideoView.Key.NONE
            return ret_val

    # Public methods

    def __init__(self, show_fps: bool = True) -> None:
        self.__fps_renderer = FPSRenderer() if show_fps else None

    def display(self) -> None:
        cv2.namedWindow(Config.WINDOW_NAME)
        cv2.moveWindow(Config.WINDOW_NAME, 200, 150)

    def close(self) -> None:
        cv2.destroyWindow(Config.WINDOW_NAME)

    def get_key(self) -> Key:
        return VideoView.Key.from_int(cv2.waitKey(1))

    def render(self, frame: np.array) -> None:
        frame = img.resized(frame, Config.VIDEO_SIZE)
        if self.__fps_renderer is not None:
            self.__fps_renderer.render(frame)
        cv2.imshow(Config.WINDOW_NAME, frame)
