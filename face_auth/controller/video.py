import numpy as np

from face_auth.model import img
from face_auth.model.input import WebcamStream
from face_auth.view.video import VideoView


class VideoController:

    # Public

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        self._view = view
        self._input_stream = input_stream

    def __enter__(self):
        self._view.display()
        self._input_stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._input_stream.stop()
        self._view.close()

    def run_loop(self) -> None:
        while True:
            key = self._view.get_key()
            frame = self._input_stream.get_frame()

            if frame is not None:
                frame = self.__preprocess_frame(frame)
                frame = self._process_frame(frame, key)
                self._view.render(frame)

            if key == VideoView.Key.ESC:
                break

    # Private

    def __preprocess_frame(self, frame: np.array) -> np.array:
        for step in [img.cropped_to_square, img.flipped_horizontally]:
            frame = step(frame)
        return frame

    # Override

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        raise NotImplementedError
