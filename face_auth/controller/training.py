import numpy as np

from .video import VideoController
from face_auth import config
from face_auth.model.detector import FaceDetector
from face_auth.model.input import WebcamStream
from face_auth.model.recognition import FaceRecognitionModel
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class TrainingController(VideoController):

    # Public methods

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(TrainingController, self).__init__(view, input_stream)
        self.__detector = FaceDetector()
        self.__face_model = FaceRecognitionModel()

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        face = self.__detector.detect_main_face(frame)

        if face is not None:
            if key == VideoView.Key.SPACE:
                self.__face_model.add_sample(frame, face.landmarks)
            elif key == VideoView.Key.ENTER:
                self.__face_model.train(config.Paths.FACE_RECOGNITION_MODEL)

            geometry_renderer.draw_landmarks(frame, face.landmarks)

        return frame
