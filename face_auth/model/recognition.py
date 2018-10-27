import cv2.cv2 as cv2
import json
import numpy as np
import sys
from enum import Enum
from itertools import islice, tee
from typing import Iterable, NamedTuple, Optional

from face_auth import config
from . import img
from .dataset import Dataset
from .detector import FaceDetector
from .geometry import Landmarks, Size
from .process import Pipeline, Step


class FaceSample(NamedTuple):
    image: np.array
    landmarks: Landmarks


class FaceRecognizer:

    class Algo(Enum):
        EIGEN = 0
        FISHER = 1
        LBPH = 2

    class ConfigKey:
        ALGO = 'algo'
        THRESH = 'thresh'

    @classmethod
    def create(cls, algo: Optional[Algo] = None) -> 'FaceRecognizer':
        if algo is None:
            algo = FaceRecognizer.Algo[config.Recognizer.ALGORITHM]

        if algo == FaceRecognizer.Algo.EIGEN:
            return EigenRecognizer()
        elif algo == FaceRecognizer.Algo.FISHER:
            return FisherRecognizer()
        else:
            return LBPHRecognizer()

    @classmethod
    def from_file(cls, model_path: str, config_path: str) -> 'FaceRecognizer':
        with open(config_path, mode='r') as config_file:
            cfg = json.load(config_file)

        algo = cls.Algo[cfg[cls.ConfigKey.ALGO]]

        recognizer = cls.create(algo)
        recognizer.threshold = float(cfg[cls.ConfigKey.THRESH])
        recognizer._load(model_path)

        return recognizer

    def __init__(self) -> None:
        self.threshold = 0.0

    def confidence_of_prediction(self, sample: FaceSample) -> float:
        confidence = self._predict(FaceRecognizer.extract_face(sample, config.DEBUG))

        if config.DEBUG:
            print('Prediction confidence: {:.2f} (threshold: {:.2f})'.format(confidence,
                                                                             self.threshold))
        return confidence

    def predict(self, sample: FaceSample) -> bool:
        return self.confidence_of_prediction(sample) < self.threshold

    def train(self, ground_truth: FaceSample, samples: [FaceSample],
              detector: FaceDetector, dataset: Dataset) -> None:

        ground_truth = self.extract_face(ground_truth, config.DEBUG)
        positives = [self.extract_face(s) for s in samples]
        negatives = dataset.training_samples(lambda x: self.extract_frontal_face(x, detector),
                                             max_samples=config.Recognizer.MAX_SAMPLES)

        n1, n2 = tee(negatives, 2)

        self._train(positives, n1)
        self.__learn_threshold(ground_truth, n2)

    def save(self, model_path: str, config_path: str) -> None:
        self._save(model_path)

        with open(config_path, mode='w') as config_file:
            cfg = {
                self.ConfigKey.ALGO: self.algo().name,
                self.ConfigKey.THRESH: '{:.2f}'.format(self.threshold)
            }
            json.dump(cfg, config_file)

    @classmethod
    def extract_frontal_face(cls, image: np.array, detector: FaceDetector,
                             debug: bool = False) -> Optional[np.array]:
        face = detector.detect_main_face(image)

        if face is None or not face.landmarks.pose_is_frontal():
            return None

        return cls.extract_face(FaceSample(image, face.landmarks), debug)

    @classmethod
    def extract_face(cls, sample: FaceSample, debug: bool = False) -> np.array:
        rect = sample.landmarks.square()
        shape = sample.landmarks.outer_shape
        matrix = sample.landmarks.alignment_matrix()

        return Pipeline.execute('Face extraction', sample.image, debug, [
            Step('To grayscale', img.to_grayscale),
            Step('Mask', lambda f: img.masked_to_shape(f, shape)),
            Step('Align', lambda f: img.transform(f, matrix, rect.size)),
            Step('Resize', lambda f: img.resized(f, Size(100, 100))),
            Step('Denoise', img.denoised),
            Step('Equalize', img.equalized),
            Step('Normalize', img.normalized)
        ])

    # Override

    def algo(self) -> Algo:
        raise NotImplementedError

    def _predict(self, image: np.array) -> float:
        raise NotImplementedError

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        raise NotImplementedError

    def _load(self, model_path: str) -> None:
        raise NotImplementedError

    def _save(self, model_path: str) -> None:
        raise NotImplementedError

    # Private

    def __learn_threshold(self, ground_truth: np.array, negatives: Iterable[np.array]) -> None:
        min_confidence = sys.maxsize
        ground_confidence = self._predict(ground_truth)

        for image in negatives:
            confidence = self._predict(image)

            if confidence < min_confidence:
                min_confidence = confidence

        self.threshold = (min_confidence + ground_confidence) / 2

        if config.DEBUG:
            print('Ground confidence: {:.2f}'.format(ground_confidence))
            print('Best confidence in training set: {:.2f}'.format(min_confidence))
            print('Learned treshold: {:.2f}'.format(self.threshold))


class EigenRecognizer(FaceRecognizer):

    def __init__(self) -> None:
        super().__init__()
        self.__rec: cv2.face_EigenFaceRecognizer = None

    # Overrides

    def algo(self) -> FaceRecognizer.Algo:
        return FaceRecognizer.Algo.EIGEN

    def _predict(self, image: np.array) -> float:
        label, confidence = self.__rec.predict(image)
        return confidence if label == 0 else float('inf')

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused

        np_array = np.asarray(positive_samples)
        self.__rec = cv2.face.EigenFaceRecognizer_create()
        self.__rec.train(np_array, np.zeros(len(np_array), dtype=np.int))

    def _load(self, model_path: str) -> None:
        self.__rec = cv2.face.EigenFaceRecognizer_create()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)


class FisherRecognizer(FaceRecognizer):

    def __init__(self) -> None:
        super().__init__()
        self.__rec: cv2.face_FisherFaceRecognizer = None

    # Overrides

    def algo(self) -> FaceRecognizer.Algo:
        return FaceRecognizer.Algo.FISHER

    def _predict(self, image: np.array) -> float:
        label, confidence = self.__rec.predict(image)
        return confidence if label == 0 else float('inf')

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:

        positives = np.asarray(list(positive_samples), dtype=np.int)
        negatives = np.asarray(list(islice(negative_samples, len(positives))), dtype=np.int)
        labels = ([0] * len(positives)) + ([1] * len(negatives))

        samples = np.concatenate((positives, negatives))
        labels = np.asarray(labels, dtype=np.int)

        self.__rec = cv2.face.FisherFaceRecognizer_create()
        self.__rec.train(samples, labels)

    def _load(self, model_path: str) -> None:
        self.__rec = cv2.face.FisherFaceRecognizer_create()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)


class LBPHRecognizer(FaceRecognizer):

    def __init__(self) -> None:
        super().__init__()
        self.__rec: cv2.face_LBPHFaceRecognizer = None

    # Overrides

    def algo(self) -> FaceRecognizer.Algo:
        return FaceRecognizer.Algo.LBPH

    def _predict(self, image: np.array) -> float:
        label, confidence = self.__rec.predict(image)
        return confidence if label == 0 else float('inf')

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused

        np_array = np.asarray(positive_samples)
        self.__rec = cv2.face.LBPHFaceRecognizer_create()
        self.__rec.train(np_array, np.zeros(len(np_array), dtype=np.int))

    def _load(self, model_path: str) -> None:
        self.__rec = cv2.face.LBPHFaceRecognizer_create()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)
