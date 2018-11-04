import numpy as np
from threading import Thread
from typing import List

from .video import VideoController
from face_auth import config
from face_auth.model.dataset import Dataset
from face_auth.model.detector import FaceDetector
from face_auth.model.input import WebcamStream
from face_auth.model.recognition import FaceRecognizer, FaceSample
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class LabelText:
    EMPTY_STATUS = 'Space: get sample | Esc: exit'
    READY_STATUS = 'Space: get sample | Enter: train | Esc: exit'
    TRAINING = 'Training, please wait...'
    DONE = 'Model trained successfully'
    SAMPLES = 'Samples: {}'


class TrainingController(VideoController):

    # Public

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(TrainingController, self).__init__(view, input_stream)

        self.__samples: List[FaceSample] = []
        self.__detector = FaceDetector()
        self.__recognizer = FaceRecognizer.create()
        self.__dataset = Dataset(config.Paths.DATASET_DIR, config.Paths.TRAINING_SET_FILE)
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
                self.__train_async(FaceSample(frame, face))

            geometry_renderer.draw_landmarks(frame, face.landmarks)

        return frame

    # Private

    def __add_sample(self, sample: FaceSample) -> None:
        self.__samples.append(sample)
        self._view.bottom_text = LabelText.READY_STATUS

    def __train(self, ground_truth: FaceSample) -> None:
        self.__is_training = True
        self._view.bottom_text = LabelText.TRAINING

        self.__recognizer.train(ground_truth, self.__samples,
                                FaceDetector(scale_factor=1), self.__dataset)
        self.__recognizer.save(config.Paths.FACE_RECOGNITION_MODEL,
                               config.Paths.FACE_RECOGNITION_MODEL_CONFIG)
        self.__samples = []

        self._view.bottom_text = LabelText.EMPTY_STATUS
        self._view.temp_message(LabelText.DONE, 3)
        self.__is_training = False

    def __train_async(self, ground_truth: FaceSample) -> None:
        if len(self.__samples) > 0:
            self.__train_thread = Thread(name='Model training',
                                         target=self.__train,
                                         args=[ground_truth],
                                         daemon=True)
            self.__train_thread.start()
