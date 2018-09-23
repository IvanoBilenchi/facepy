import cv2.cv2 as cv2
import numpy as np
from typing import Callable

from .fps_renderer import FPSRenderer
from face_auth.model import img


class Webcam:
    """Models the webcam."""

    # Public methods

    def __init__(self, window_name: str = 'Webcam',
                 frame_handler: Callable[[np.array, int], None] = None) -> None:
        self.window_name = window_name
        self.__frame_handler = frame_handler
        self.__cam: cv2.VideoCapture = None
        self.__fps_renderer: FPSRenderer = None

    def __enter__(self):
        cv2.namedWindow(self.window_name)
        cv2.moveWindow(self.window_name, 100, 80)
        self.__cam = cv2.VideoCapture(0)
        self.__fps_renderer = FPSRenderer(self.__cam)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cam.release()
        cv2.destroyAllWindows()

    def start_capture(self) -> None:
        while True:
            key = cv2.waitKey(1)
            frame = self.__get_frame()

            if frame is not None:
                self.__frame_handler(frame, key)
                frame = self.__postprocess_frame(frame)
                self.__display_frame(frame)

            if key == 27:
                break

    # Private methods

    def __get_frame(self) -> np.array:
        success, frame = self.__cam.read()

        if not success:
            return None

        frame = cv2.flip(frame, 1)
        frame = img.cropped_to_square(frame)
        self.__fps_renderer.frame_tick()

        return frame

    def __display_frame(self, frame: np.array) -> None:
        cv2.imshow(self.window_name, frame)

    def __postprocess_frame(self, frame: np.array) -> np.array:
        self.__fps_renderer.render(frame)
        return frame
