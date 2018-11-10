import numpy as np
from threading import Thread
from time import sleep

from .video import VideoController
from face_auth import config
from face_auth.model.detector import VideoFaceDetector
from face_auth.model.geometry import Face
from face_auth.model.input import WebcamStream
from face_auth.model.recognition import FaceRecognizer, FaceSample
from face_auth.view import color, geometry_renderer
from face_auth.view.video import VideoView


class LabelText:
    READY_STATUS = 'Esc: exit'
    AUTHENTICATED = 'Authenticated'
    NOT_AUTHENTICATED = 'Not authenticated'
    NO_DETECTION = 'No face detected'


class AuthenticationController(VideoController):

    # Public

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(AuthenticationController, self).__init__(view, input_stream)
        self.__detector = VideoFaceDetector()
        self.__recognizer = FaceRecognizer.from_file(config.Paths.FACE_RECOGNITION_MODEL,
                                                     config.Paths.FACE_RECOGNITION_MODEL_CONFIG)

        self.__sample: FaceSample = None
        self.__authenticated = False
        self.__start_authenticating()

        view.bottom_text = LabelText.READY_STATUS

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        face = self.__detector.detect_main_face(frame)

        if face is not None:
            self.__submit_sample(frame, face)
            geometry_renderer.draw_landmarks(frame, face.landmarks,
                                             draw_bg=True, color=self.__landmarks_color())

        return frame

    # Private

    def __submit_sample(self, frame: np.array, face: Face) -> None:
        if self.__sample is None:
            self.__sample = FaceSample(frame, face)

    def __landmarks_color(self) -> (int, int, int):
        return color.GREEN if self.__authenticated else color.RED

    def __start_authenticating(self) -> None:
        self.__auth_thread = Thread(name='Authentication thread',
                                    target=self.__authenticate_loop,
                                    daemon=True)
        self.__auth_thread.start()

    def __authenticate_loop(self) -> None:
        interval = 1.0

        while True:
            if self.__sample:
                self.__authenticated = self.__recognizer.predict(self.__sample)
                self.__sample = None

                if self.__authenticated:
                    msg = LabelText.AUTHENTICATED
                else:
                    msg = LabelText.NOT_AUTHENTICATED

                self._view.top_text = msg
            else:
                self._view.top_text = LabelText.NO_DETECTION

            sleep(interval)
