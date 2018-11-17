import numpy as np
from threading import Thread
from typing import List

from .video import VideoController
from face_auth import config
from face_auth.model.detector import StaticFaceDetector, VideoFaceDetector
from face_auth.model.input import WebcamStream
from face_auth.model.verification import FaceVerifier, FaceSample
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class LabelText:
    EMPTY_STATUS = 'Space: get sample | Esc: exit'
    READY_STATUS = 'Space: get sample | Enter: train | Esc: exit'
    TRAINING = 'Training, please wait...'
    DONE = 'Model trained successfully'
    SAMPLES = 'Samples: {}'


class TrainVerifierVideoController(VideoController):

    # Public

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(TrainVerifierVideoController, self).__init__(view, input_stream)

        self.__samples: List[FaceSample] = []
        self.__detector = VideoFaceDetector()
        self.__verifier = FaceVerifier.create()
        self.__is_training = False

        view.bottom_text = LabelText.EMPTY_STATUS

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        if self.__is_training:
            return frame

        self._view.top_text = LabelText.SAMPLES.format(len(self.__samples))
        face = self.__detector.detect_main_face(frame)

        if face is not None:
            if key == VideoView.Key.SPACE:
                self.__add_sample(FaceSample(frame, face))
            elif key == VideoView.Key.ENTER:
                self.__train_async()

            geometry_renderer.draw_landmarks(frame, face.landmarks)

        return frame

    # Private

    def __add_sample(self, sample: FaceSample) -> None:
        self.__samples.append(sample)

        if len(self.__samples) > 1:
            self._view.bottom_text = LabelText.READY_STATUS

    def __train(self) -> None:
        self.__is_training = True
        self._view.bottom_text = LabelText.TRAINING

        self.__verifier.train(self.__samples, StaticFaceDetector(scale_factor=1))
        self.__verifier.save(config.Paths.VERIFICATION_MODEL,
                             config.Paths.VERIFICATION_MODEL_CONFIG)
        self.__samples = []

        self._view.bottom_text = LabelText.EMPTY_STATUS
        self._view.temp_message(LabelText.DONE, 3)
        self.__is_training = False

    def __train_async(self) -> None:
        if len(self.__samples) > 0:
            self.__train_thread = Thread(name='Model training',
                                         target=self.__train,
                                         daemon=True)
            self.__train_thread.start()
