import numpy as np

from facepy.model import img
from facepy.model.input import WebcamStream
from facepy.view.video import VideoView


class VideoController:

    # Public

    def __init__(self) -> None:
        self._view = VideoView()
        self._input_stream = WebcamStream()

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
