import numpy as np
from threading import Thread
from time import sleep

from .video import VideoController
from facepy.model.detector import VideoFaceDetector
from facepy.model.geometry import Face
from facepy.model.verification import FaceVerifier, FaceSample
from facepy.view import color, geometry_renderer
from facepy.view.video import VideoView


class LabelText:
    READY_STATUS = 'Esc: exit'
    VERIFIED = 'Verified'
    NOT_VERIFIED = 'Not verified'
    NO_DETECTION = 'No face detected'


class VerificationVideoController(VideoController):

    # Public

    def __init__(self, model_dir: str) -> None:
        super(VerificationVideoController, self).__init__()
        self.__detector = VideoFaceDetector()
        self.__verifier = FaceVerifier.from_dir(model_dir)

        self.__sample: FaceSample = None
        self.__verified = False
        self.__start_verifying()
        self.__verification_thread: Thread = None

        self._view.bottom_text = LabelText.READY_STATUS

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
        return color.GREEN if self.__verified else color.RED

    def __start_verifying(self) -> None:
        self.__verification_thread = Thread(name='Verification thread',
                                            target=self.__verification_loop,
                                            daemon=True)
        self.__verification_thread.start()

    def __verification_loop(self) -> None:
        interval = 1.0

        while True:
            if self.__sample:
                self.__verified = self.__verifier.predict_sample(self.__sample)
                self.__sample = None

                if self.__verified:
                    msg = LabelText.VERIFIED
                else:
                    msg = LabelText.NOT_VERIFIED

                self._view.top_text = msg
            else:
                self._view.top_text = LabelText.NO_DETECTION

            sleep(interval)
