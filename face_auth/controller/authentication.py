import numpy as np

from .video import VideoController
from face_auth import config
from face_auth.model import detector
from face_auth.model.input import WebcamStream
from face_auth.model.recognition import FaceRecognizer
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class AuthenticationController(VideoController):

    # Public methods

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(AuthenticationController, self).__init__(view, input_stream)
        self.__recognizer = FaceRecognizer(config.Paths.FACE_RECOGNITION_MODEL)

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        face = detector.detect_main_face(frame)

        if face is not None:
            if key == VideoView.Key.SPACE:
                confidence = self.__recognizer.confidence_of_prediction(frame, face.landmarks)
                print('Confidence: {}'.format(confidence))

            geometry_renderer.draw_landmarks(frame, face.landmarks)

        return frame
