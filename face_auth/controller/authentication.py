import numpy as np

from .video import VideoController
from face_auth import config
from face_auth.model.detector import FaceDetector
from face_auth.model.input import WebcamStream
from face_auth.model.recognition import FaceRecognizer, FaceSample
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class AuthenticationController(VideoController):

    # Public methods

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(AuthenticationController, self).__init__(view, input_stream)
        self.__detector = FaceDetector()
        self.__recognizer = FaceRecognizer.from_file(config.Paths.FACE_RECOGNITION_MODEL,
                                                     config.Paths.FACE_RECOGNITION_MODEL_CONFIG)

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        face = self.__detector.detect_main_face(frame)

        if face is not None:
            if key == VideoView.Key.SPACE:
                sample = FaceSample(frame, face)
                outcome = self.__recognizer.predict(sample)
                print('✅ Authorized' if outcome else '⛔ Not authorized️')

            geometry_renderer.draw_landmarks(frame, face.landmarks)

        return frame
