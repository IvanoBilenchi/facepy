import numpy as np
from typing import List

from .video import VideoController
from face_auth import config
from face_auth.model.dataset import Dataset
from face_auth.model.detector import FaceDetector
from face_auth.model.input import WebcamStream
from face_auth.model.recognition import FaceRecognizer, FaceSample
from face_auth.view import geometry_renderer
from face_auth.view.video import VideoView


class TrainingController(VideoController):

    # Public methods

    def __init__(self, view: VideoView, input_stream: WebcamStream) -> None:
        super(TrainingController, self).__init__(view, input_stream)
        self.__samples: List[FaceSample] = []
        self.__detector = FaceDetector()
        self.__recognizer = FaceRecognizer()
        self.__dataset = Dataset(config.Paths.DATASET_DIR, config.Paths.TRAINING_SET_FILE)

    # Overrides

    def _process_frame(self, frame: np.array, key: VideoView.Key) -> np.array:
        face = self.__detector.detect_main_face(frame)

        if face is not None:
            if key == VideoView.Key.SPACE:
                self.__samples.append(FaceSample(frame, face.landmarks))
            elif key == VideoView.Key.ENTER:
                self.__recognizer.train(FaceSample(frame, face.landmarks), self.__samples,
                                        FaceDetector(scale_factor=1), self.__dataset)
                self.__recognizer.save(config.Paths.FACE_RECOGNITION_MODEL,
                                       config.Paths.FACE_RECOGNITION_MODEL_CONFIG)

            geometry_renderer.draw_landmarks(frame, face.landmarks)

        return frame
