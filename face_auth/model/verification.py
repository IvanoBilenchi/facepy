import cv2.cv2 as cv2
import dlib
import numpy as np
import sys
from enum import Enum
from itertools import islice, tee
from typing import Iterable, List, Optional

from face_auth import config
from . import fileutils, img
from .dataset import Dataset, DataSample
from .detector import Face, StaticFaceDetector
from .geometry import Landmarks, Point, Size
from .process import Pipeline, Step


class FaceSample:

    def __init__(self, image: np.array, face: Face) -> None:
        self.image = image.copy()
        self.face = face


class FaceVerifier:

    class Algo(Enum):
        EIGEN = 0
        FISHER = 1
        LBPH = 2
        EUCLIDEAN = 3
        CNN = 4

    class ConfigKey:
        ALGO = 'algo'
        THRESH = 'thresh'

    @classmethod
    def create(cls, algo: Optional[Algo] = None) -> 'FaceVerifier':
        if algo is None:
            algo = FaceVerifier.Algo[config.Recognizer.ALGORITHM]

        if algo == FaceVerifier.Algo.EIGEN:
            return EigenVerifier()
        elif algo == FaceVerifier.Algo.FISHER:
            return FisherVerifier()
        elif algo == FaceVerifier.Algo.LBPH:
            return LBPHVerifier()
        elif algo == FaceVerifier.Algo.EUCLIDEAN:
            return EuclideanVerifier()
        else:
            return CNNVerifier()

    @classmethod
    def from_file(cls, model_path: str, config_path: str) -> 'FaceVerifier':
        cfg = fileutils.load_json(config_path)

        algo = cls.Algo[cfg[cls.ConfigKey.ALGO]]
        verifier = cls.create(algo)
        verifier.threshold = float(cfg[cls.ConfigKey.THRESH])
        verifier._load(model_path)

        return verifier

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
              detector: StaticFaceDetector, dataset: Dataset) -> None:

        ground_truth = self.__extract_face(ground_truth, config.DEBUG)
        positives = [self.__extract_face(s) for s in samples]

        def preprocessor(sample: DataSample) -> Optional[DataSample]:
            image = self.__extract_frontal_face(sample.image, detector)
            new_sample = None

            if image is not None:
                new_sample = DataSample(sample.file_path)
                new_sample.image = image

            return new_sample

        negatives = dataset.negative_verification_images(sample_filter=preprocessor,
                                                         max_samples=config.Recognizer.MAX_SAMPLES)

        n1, n2 = tee(negatives, 2)

        self._train(positives, n1)
        self.__learn_threshold(ground_truth, n2)

    def save(self, model_path: str, config_path: str) -> None:
        fileutils.create_parent_dir(model_path)
        self._save(model_path)

        cfg = {
            self.ConfigKey.ALGO: self.algo().name,
            self.ConfigKey.THRESH: '{:.5f}'.format(self.threshold)
        }

        fileutils.create_parent_dir(config_path)
        fileutils.save_json(cfg, config_path)

    # Must override

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

    def __extract_frontal_face(self, image: np.array, detector: StaticFaceDetector,
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


class EigenVerifier(FaceVerifier):

    def __init__(self) -> None:
        super().__init__()
        self.__rec: cv2.face_EigenFaceRecognizer = None

    # Overrides

    def algo(self) -> FaceVerifier.Algo:
        return FaceVerifier.Algo.EIGEN

    def _predict(self, image: np.array) -> float:
        label, confidence = self.__rec.predict(image)
        return confidence if label == 0 else float('inf')

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused

        np_array = np.asarray(list(positive_samples))
        self.__rec = cv2.face.EigenFaceRecognizer_create()
        self.__rec.train(np_array, np.zeros(len(np_array), dtype=np.int))

    def _load(self, model_path: str) -> None:
        self.__rec = cv2.face.EigenFaceRecognizer_create()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)


class FisherVerifier(FaceVerifier):

    def __init__(self) -> None:
        super().__init__()
        self.__rec: cv2.face_FisherFaceRecognizer = None

    # Overrides

    def algo(self) -> FaceVerifier.Algo:
        return FaceVerifier.Algo.FISHER

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


class LBPHVerifier(FaceVerifier):

    def __init__(self) -> None:
        super().__init__()
        self.__rec: cv2.face_LBPHFaceRecognizer = None

    # Overrides

    def algo(self) -> FaceVerifier.Algo:
        return FaceVerifier.Algo.LBPH

    def _predict(self, image: np.array) -> float:
        label, confidence = self.__rec.predict(image)
        return confidence if label == 0 else float('inf')

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused

        np_array = np.asarray(list(positive_samples))
        self.__rec = cv2.face.LBPHFaceRecognizer_create()
        self.__rec.train(np_array, np.zeros(len(np_array), dtype=np.int))

    def _load(self, model_path: str) -> None:
        self.__rec = cv2.face.LBPHFaceRecognizer_create()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)


class EmbeddingsVerifier(FaceVerifier):

    def __init__(self) -> None:
        super().__init__()
        self.__embeddings = []

    # Must override

    def algo(self) -> FaceVerifier.Algo:
        raise NotImplementedError

    def _compute_embedding(self, image: np.array) -> np.array:
        raise NotImplementedError

    # Overrides

    def needs_preprocessing(self) -> bool:
        return False

    def _predict(self, image: np.array) -> float:
        sample = self._compute_embedding(image)

        min_distance = float('inf')

        if sample is None:
            return min_distance

        for embedding in self.__embeddings:
            distance = self.__distance(embedding, sample)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused
        self.__embeddings = [self._compute_embedding(i) for i in positive_samples]

    def _load(self, model_path: str) -> None:
        json_array = fileutils.load_json(model_path)
        if isinstance(json_array, list):
            self.__embeddings = [np.asarray(e) for e in json_array if isinstance(e, list)]

    def _save(self, model_path: str) -> None:
        embeddings = [e.tolist() for e in self.__embeddings]
        fileutils.save_json(embeddings, model_path)

    # Private

    def __distance(self, embedding: np.array, other: np.array) -> float:
        return np.linalg.norm(embedding - other)


class EuclideanVerifier(EmbeddingsVerifier):

    def __init__(self) -> None:
        super().__init__()
        self.__detector = StaticFaceDetector(scale_factor=1)

    # Overrides

    def algo(self) -> FaceVerifier.Algo:
        return FaceVerifier.Algo.EUCLIDEAN

    def _compute_embedding(self, image: np.array) -> np.array:
        face = self.__detector.detect_main_face(image)

        if face is None:
            return None

        traits = self.__rec_traits(face.landmarks)
        n_traits = len(traits)
        norm_factor = face.landmarks.left_eye[0].distance(face.landmarks.right_eye[3])

        embedding = np.zeros(self.__num_combinations_two(n_traits))
        idx = 0

        for i in range(n_traits - 1):
            for j in range(i + 1, n_traits):
                embedding[idx] = traits[i].distance(traits[j]) / norm_factor
                idx += 1

        return embedding

    # Private

    def __rec_traits(self, lm: Landmarks) -> List[Point]:
        return [
            lm.left_eye[0], lm.left_eye[3], lm.right_eye[0], lm.right_eye[3],
            lm.nose_bridge[0], lm.nose_tip[2], lm.nose_tip[0], lm.nose_tip[-1],
            lm.top_lip[0], lm.top_lip[2], lm.top_lip[4], lm.top_lip[6], lm.bottom_lip[3]
        ]

    def __num_combinations_two(self, count: int) -> int:
        return count * (count - 1) // 2

    def __distance(self, embedding: np.array, other: np.array) -> float:
        return np.linalg.norm(embedding - other)


class CNNVerifier(EmbeddingsVerifier):

    def __init__(self) -> None:
        super().__init__()
        self.__detector = dlib.cnn_face_detection_model_v1(config.Paths.CNN_FACE_DETECTOR_MODEL)
        self.__predictor = dlib.shape_predictor(config.Paths.FACE_LANDMARKS_SMALL_MODEL)
        self.__net = dlib.face_recognition_model_v1(config.Paths.CNN_FACE_DESCRIPTOR_MODEL)

    # Overrides

    def algo(self) -> FaceVerifier.Algo:
        return FaceVerifier.Algo.CNN

    def _compute_embedding(self, image: np.array) -> np.array:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rect = self.__detector(image)[0].rect
        landmarks = self.__predictor(image, rect)
        return np.array(self.__net.compute_face_descriptor(image, landmarks))
