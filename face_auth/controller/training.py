import cv2.cv2 as cv2
import numpy as np

from face_auth.model import detector, img
from face_auth.model.input import WebcamStream
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class TrainingController:

    # Public methods

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        self.__view = view
        self.__input_stream = input_stream

    def __enter__(self):
        self.__view.display()
        self.__input_stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__input_stream.stop()
        self.__view.close()

    def run_loop(self) -> None:
        while True:
            key = VideoView.Key.NONE
            frame = self.__input_stream.get_frame()

            if frame is not None:
                frame = self.__process_frame(frame)
                self.__view.render(frame)
                key = self.__view.get_key()

            if key == VideoView.Key.ESC:
                break

    # Private methods

    @staticmethod
    def __process_frame(frame: np.array) -> np.array:
        frame = cv2.flip(frame, 1)
        frame = img.cropped_to_square(frame)

        faces = detector.detect_faces(frame)

        for face in faces:
            geometry_renderer.draw_rect(frame, face)
            landmarks = detector.detect_landmarks(frame, face)
            geometry_renderer.draw_landmarks(frame, landmarks)

        return frame
