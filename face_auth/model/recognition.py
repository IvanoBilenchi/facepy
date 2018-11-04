import cv2.cv2 as cv2
import dlib
import json
import numpy as np
import sys
from enum import Enum
from itertools import islice, tee
from typing import Iterable, Optional

from face_auth import config
from . import img
from .dataset import Dataset
from .detector import Face, FaceDetector
from .geometry import Size
from .process import Pipeline, Step


class FaceSample:

    def __init__(self, image: np.array, face: Face) -> None:
        self.image = image.copy()
        self.face = face


class FaceRecognizer:

    class Algo(Enum):
        EIGEN = 0
        FISHER = 1
        LBPH = 2
        CNN = 3

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
        elif algo == FaceRecognizer.Algo.LBPH:
            return LBPHRecognizer()
        else:
            return CNNRecognizer()

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
        image = self.__extract_face(sample, config.DEBUG)
        confidence = self._predict(image)

        if config.DEBUG:
            print('Prediction confidence: {:.2f} (threshold: {:.2f})'.format(confidence,
                                                                             self.threshold))
        return confidence

    def predict(self, sample: FaceSample) -> bool:
        return self.confidence_of_prediction(sample) < self.threshold

    def train(self, ground_truth: FaceSample, samples: [FaceSample],
              detector: FaceDetector, dataset: Dataset) -> None:

        ground_truth = self.__extract_face(ground_truth, config.DEBUG)
        positives = [self.__extract_face(s) for s in samples]
        negatives = dataset.training_samples(lambda x: self.__extract_frontal_face(x, detector),
                                             max_samples=config.Recognizer.MAX_SAMPLES)

        n1, n2 = tee(negatives, 2)

        self._train(positives, n1)
        self.__learn_threshold(ground_truth, n2)

    def save(self, model_path: str, config_path: str) -> None:
        self._save(model_path)

        with open(config_path, mode='w') as config_file:
            cfg = {
                self.ConfigKey.ALGO: self.algo().name,
                self.ConfigKey.THRESH: '{:.5f}'.format(self.threshold)
            }
            json.dump(cfg, config_file)

    # Override

    def algo(self) -> Algo:
        raise NotImplementedError

    def needs_preprocessing(self) -> bool:
        return True

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

    def __extract_frontal_face(self, image: np.array, detector: FaceDetector,
                               debug: bool = False) -> Optional[np.array]:
        face = detector.detect_main_face(image)

        if face is None or not face.landmarks.pose_is_frontal():
            return None

        return self.__extract_face(FaceSample(image, face), debug)

    def __extract_face(self, sample: FaceSample, debug: bool = False) -> np.array:

        if self.needs_preprocessing():
            rect = sample.face.landmarks.square()
            shape = sample.face.landmarks.thin_shape
            matrix = sample.face.landmarks.alignment_matrix()

            steps = [
                Step('To grayscale', img.to_grayscale),
                Step('Mask', lambda f: img.masked_to_shape(f, shape)),
                Step('Align', lambda f: img.transform(f, matrix, rect.size)),
                Step('Resize', lambda f: img.resized(f, Size(100, 100))),
                Step('Denoise', img.denoised),
                Step('Equalize', img.equalized),
                Step('Normalize', img.normalized)
            ]
        else:
            steps = [Step('Resize', lambda f: img.resized(f, Size(250, 250)))]

        return Pipeline.execute('Face extraction', sample.image, debug, steps)


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


class CNNRecognizer(FaceRecognizer):

    def __init__(self) -> None:
        super().__init__()
        self.__embeddings = []
        self.__detector = dlib.cnn_face_detection_model_v1(config.Paths.CNN_FACE_DETECTOR_MODEL)
        self.__predictor = dlib.shape_predictor(config.Paths.FACE_LANDMARKS_SMALL_MODEL)
        self.__net = dlib.face_recognition_model_v1(config.Paths.CNN_FACE_DESCRIPTOR_MODEL)

    # Overrides

    def algo(self) -> FaceRecognizer.Algo:
        return FaceRecognizer.Algo.CNN

    def needs_preprocessing(self) -> bool:
        return False

    def _predict(self, image: np.array) -> float:
        sample = self.__compute_embedding(image)
        min_distance = float('inf')

        for embedding in self.__embeddings:
            distance = self.__distance(embedding, sample)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused
        self.__embeddings = [self.__compute_embedding(i) for i in positive_samples]

    def _load(self, model_path: str) -> None:
        with open(model_path, mode='r') as model_file:
            json_array = json.load(model_file)

            if isinstance(json_array, list):
                self.__embeddings = [np.asarray(e) for e in json_array if isinstance(e, list)]

    def _save(self, model_path: str) -> None:
        with open(model_path, mode='w') as model_file:
            embeddings = [e.tolist() for e in self.__embeddings]
            json.dump(embeddings, model_file)

    # Private

    def __distance(self, embedding: np.array, other: np.array) -> float:
        return np.linalg.norm(embedding - other)

    def __compute_embedding(self, image: np.array) -> np.array:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rect = self.__detector(image)[0].rect
        landmarks = self.__predictor(image, rect)
        return np.array(self.__net.compute_face_descriptor(image, landmarks))
