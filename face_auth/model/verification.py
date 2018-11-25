import cv2.cv2 as cv2
import dlib
import math
import numpy as np
import sys
from itertools import islice, tee
from os import path
from typing import Iterable, List, Optional

from face_auth import config
from . import dataset, fileutils, img
from .dataset import DataSample
from .detector import FaceSample, StaticFaceDetector
from .geometry import Landmarks, Point, Size
from .process import Pipeline, Step
from .recognition_algo import RecognitionAlgo


class FaceVerifier:

    class ConfigKey:
        ALGO = 'algo'
        THRESH = 'thresh'
        NAME = 'name'

    class FileName:
        MODEL = 'model.dat'
        CONFIG = 'model_config.json'

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'FaceVerifier':
        if algo == RecognitionAlgo.EIGEN:
            return EigenVerifier()
        elif algo == RecognitionAlgo.FISHER:
            return FisherVerifier()
        elif algo == RecognitionAlgo.LBPH:
            return LBPHVerifier()
        elif algo == RecognitionAlgo.EUCLIDEAN:
            return EuclideanVerifier()
        else:
            return CNNVerifier()

    @classmethod
    def from_dir(cls, dir_path: str) -> 'FaceVerifier':
        model_path = path.join(dir_path, cls.FileName.MODEL)
        config_path = path.join(dir_path, cls.FileName.CONFIG)

        cfg = fileutils.load_json(config_path)
        algo = RecognitionAlgo[cfg[cls.ConfigKey.ALGO]]
        verifier = cls.create(algo)
        verifier.threshold = float(cfg[cls.ConfigKey.THRESH])
        verifier.person_name = str(cfg[cls.ConfigKey.NAME])
        verifier._load(model_path)

        return verifier

    def __init__(self, person_name: str = None) -> None:
        self.threshold = 0.0
        self.person_name = person_name
        self._detector = StaticFaceDetector(scale_factor=1)

    def confidence(self, image: np.array) -> float:
        sample = self._detector.extract_main_face_sample(image)
        return self.confidence_for_sample(sample) if sample else float('inf')

    def predict(self, image: np.array) -> bool:
        return self.confidence(image) < self.threshold

    def confidence_for_sample(self, sample: FaceSample) -> float:
        image = self.__extract_face(sample, config.DEBUG)
        confidence = self._predict(image)

        if config.DEBUG:
            print('Prediction confidence: {:.2f} (threshold: {:.2f})'.format(confidence,
                                                                             self.threshold))
        return confidence

    def predict_sample(self, sample: FaceSample) -> bool:
        return self.confidence_for_sample(sample) < self.threshold

    def train(self, samples: [FaceSample]) -> None:
        samples_count = len(samples)

        if samples_count < 2:
            raise ValueError('You need at least two samples to train a verifier.')

        samples = [self.__extract_face(s, config.DEBUG) for s in samples]
        samples_count = len(samples)

        if samples_count < 2:
            raise ValueError('Could not extract enough faces from the provided samples.')

        truths_count = math.ceil(samples_count / 5)
        positives_count = samples_count - truths_count

        positives = samples[:positives_count]
        truths = samples[-truths_count:]

        def preprocessor(sample: DataSample) -> Optional[DataSample]:
            image = self.__extract_frontal_face(sample.image)
            new_sample = None

            if image is not None:
                new_sample = DataSample(sample.file_path)
                new_sample.image = image

            return new_sample

        negatives = dataset.negative_verification_images(dataset.all_samples(preprocessor),
                                                         max_samples=config.Recognizer.VERIFICATION_MAX_SAMPLES)

        n1, n2 = tee(negatives, 2)

        self._train(positives, n1)
        self.__learn_threshold(truths, n2)

    def save(self, model_dir: str) -> None:
        fileutils.create_dir(model_dir)
        model_path = path.join(model_dir, self.FileName.MODEL)
        config_path = path.join(model_dir, self.FileName.CONFIG)

        self._save(model_path)

        cfg = {
            self.ConfigKey.NAME: self.person_name if self.person_name is not None else '',
            self.ConfigKey.ALGO: self.algo().name,
            self.ConfigKey.THRESH: '{:.5f}'.format(self.threshold)
        }

        fileutils.create_parent_dir(config_path)
        fileutils.save_json(cfg, config_path)

    # Must override

    def algo(self) -> RecognitionAlgo:
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

    def __learn_threshold(self, truths: List[np.array], negatives: Iterable[np.array]) -> None:
        avg_positive_confidence = 0

        for image in truths:
            avg_positive_confidence += self._predict(image)

        avg_positive_confidence /= len(truths)

        min_negative_confidence = sys.maxsize

        for image in negatives:
            confidence = self._predict(image)

            if confidence < min_negative_confidence:
                min_negative_confidence = confidence

        self.threshold = (min_negative_confidence + avg_positive_confidence) / 2

        if config.DEBUG:
            print('Average positive confidence: {:.2f}'.format(avg_positive_confidence))
            print('Minimum negative confidence: {:.2f}'.format(min_negative_confidence))
            print('Learned treshold: {:.2f}'.format(self.threshold))

    def __extract_frontal_face(self, image: np.array, debug: bool = False) -> Optional[np.array]:
        sample = self._detector.extract_main_face_sample(image)

        if sample is None or not sample.pose_is_frontal():
            return None

        return self.__extract_face(sample, debug)

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

    def algo(self) -> RecognitionAlgo:
        return RecognitionAlgo.EIGEN

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

    def algo(self) -> RecognitionAlgo:
        return RecognitionAlgo.FISHER

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

    def algo(self) -> RecognitionAlgo:
        return RecognitionAlgo.LBPH

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

    def algo(self) -> RecognitionAlgo:
        raise NotImplementedError

    def _compute_embedding(self, image: np.array) -> Optional[np.array]:
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
        embeddings = (self._compute_embedding(i) for i in positive_samples)
        self.__embeddings = [e for e in embeddings if e is not None]

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

    # Overrides

    def algo(self) -> RecognitionAlgo:
        return RecognitionAlgo.EUCLIDEAN

    def _compute_embedding(self, image: np.array) -> Optional[np.array]:
        face = self._detector.detect_main_face(image)

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

    def algo(self) -> RecognitionAlgo:
        return RecognitionAlgo.CNN

    def _compute_embedding(self, image: np.array) -> Optional[np.array]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.__detector(image)
        embedding = None

        if detections is not None and len(detections) > 0:
            rect = detections[0].rect
            landmarks = self.__predictor(image, rect)
            embedding = np.array(self.__net.compute_face_descriptor(image, landmarks))

        return embedding
