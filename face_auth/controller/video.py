import numpy as np

from face_auth.model import img
from face_auth.model.input import WebcamStream
from face_auth.view.video import VideoView


class VideoController:

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
            key = self.__view.get_key()
            frame = self.__input_stream.get_frame()

            if frame is not None:
                frame = self.__preprocess_frame(frame)
                frame = self._process_frame(frame, key)
                self.__view.render(frame)

            if key == VideoView.Key.ESC:
                break

    # Private methods

    def __preprocess_frame(self, frame: np.array) -> np.array:
        for step in [img.cropped_to_square, img.flipped_horizontally]:
            frame = step(frame)
        return frame

    # Override

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        raise NotImplementedError
