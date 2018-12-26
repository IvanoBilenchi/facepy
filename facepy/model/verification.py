import math
from itertools import islice, tee
from os import path
from typing import Iterable, List, Optional

import cv2.cv2 as cv2
import numpy as np

from facepy import config
from . import dataset, fileutils, img, preprocess
from .dataset import DataSample
from .detector import FaceSample, StaticFaceDetector
from .feature_extractor import CNNFeatureExtractor, FeatureExtractor, GeometricFeatureExtractor
from .recognition_algo import RecognitionAlgo


class FaceVerifier:
    """
    Abstract face verifier class.

    A face verifier works by computing a 'confidence' score for a certain image,
    which is then thresholded in order to determine if the image depicts the person
    that the verifier was trained to recognize.

    Value of the threshold can be set arbitrarily depending on the required performance
    in terms of false positives and/or negatives. This class attempts to automatically learn
    a suitable threshold value by running the verifier on subsets of positive and negative
    sample images (see __learn_threshold()).
    """

    class ConfigKey:
        """Model configuration file keys."""
        ALGO = 'algo'
        THRESH = 'thresh'
        NAME = 'name'

    class FileName:
        """Model file names."""
        MODEL = 'model.dat'
        CONFIG = 'model_config.json'

    # Public

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'FaceVerifier':
        """Factory method: creates a face verifier using the specified algorithm."""
        if algo in [RecognitionAlgo.EIGEN, RecognitionAlgo.FISHER, RecognitionAlgo.LBPH]:
            return OpenCVVerifier.create(algo)
        else:
            return FeaturesVerifier.create(algo)

    @classmethod
    def from_dir(cls, dir_path: str) -> 'FaceVerifier':
        """Factory method: loads a pre-trained face verifier from the specified directory."""
        model_path = path.join(dir_path, cls.FileName.MODEL)
        config_path = path.join(dir_path, cls.FileName.CONFIG)

        cfg = fileutils.load_json(config_path)
        algo = RecognitionAlgo[cfg[cls.ConfigKey.ALGO]]
        verifier = cls.create(algo)
        verifier.threshold = float(cfg[cls.ConfigKey.THRESH])
        verifier.person_name = str(cfg[cls.ConfigKey.NAME])
        verifier._load(model_path)

        return verifier

    def __init__(self) -> None:
        self.threshold = 0.0
        self.person_name = None
        self._detector = StaticFaceDetector(scale_factor=1)

    def confidence(self, image: np.array) -> float:
        """Returns the confidence value for the specified image."""
        sample = self._detector.get_main_face_sample(image)
        return self.confidence_for_sample(sample) if sample else float('inf')

    def predict(self, image: np.array) -> bool:
        """
        Checks whether the specified image depicts the person
        that the verifier was trained to recognize.
        """
        return self.confidence(image) < self.threshold

    def confidence_for_sample(self, sample: FaceSample) -> float:
        """Returns the confidence value for the specified face sample."""
        image = preprocess.prepare_for_recognition(sample,
                                                   preprocess=self.needs_preprocessing(),
                                                   debug=config.DEBUG)
        confidence = self._predict(image)

        if config.DEBUG:
            print('Prediction confidence: {:.2f} (threshold: {:.2f})'.format(confidence,
                                                                             self.threshold))
        return confidence

    def predict_sample(self, sample: FaceSample) -> bool:
        """
        Checks whether the specified sample depicts the person
        that the verifier was trained to recognize.
        """
        return self.confidence_for_sample(sample) < self.threshold

    def train(self, samples: [FaceSample]) -> None:
        """
        Trains the verifier with the specified positive samples.

        Samples are split in two variably sized sets, whose size depends on
        VERIFICATION_POSITIVE_TRAINING_SAMPLES_SPLIT (in config.py): the first set is used
        to actually train the verifier; the second set is used to learn a suitable threshold value.
        """
        samples_count = len(samples)

        if samples_count < 2:
            raise ValueError('You need at least two samples to train a verifier.')

        should_preprocess = self.needs_preprocessing()

        samples = [
            preprocess.prepare_for_recognition(sample,
                                               preprocess=should_preprocess,
                                               debug=config.DEBUG)
            for sample in samples
        ]

        samples_count = len(samples)

        if samples_count < 2:
            raise ValueError('Could not extract enough faces from the provided samples.')

        split_ratio = config.Recognizer.VERIFICATION_POSITIVE_TRAINING_SAMPLES_SPLIT

        truths_count = math.ceil(samples_count * split_ratio)
        positives_count = samples_count - truths_count

        positives = samples[:positives_count]
        truths = samples[-truths_count:]

        def preprocessor(sample: DataSample) -> Optional[DataSample]:
            face_sample = self._detector.get_main_face_sample(sample.image)

            if face_sample is None or not face_sample.pose_is_frontal():
                return None

            image = preprocess.prepare_for_recognition(face_sample, preprocess=should_preprocess)
            new_sample = None

            if image is not None:
                new_sample = DataSample(sample.file_path)
                new_sample.image = image

            return new_sample

        def negative_verification_images(data_samples: Iterable[DataSample]) -> Iterable[np.array]:

            max_samples = config.Recognizer.VERIFICATION_NEGATIVE_TRAINING_SAMPLES
            skip_person: str = None
            n_samples = 0

            for sample in data_samples:

                sample_name = sample.person_name

                if skip_person != sample_name:
                    skip_person = None

                if sample_name == self.person_name or sample_name == skip_person:
                    continue

                skip_person = sample_name

                if n_samples < max_samples:
                    n_samples += 1
                    yield sample.image
                else:
                    break

        negatives = negative_verification_images(dataset.all_samples(preprocessor))

        if self.uses_negatives():
            n1, n2 = tee(negatives, 2)
            self._train(positives, n1)
            self.__learn_threshold(truths, n2)
        else:
            self._train(positives, [])
            self.__learn_threshold(truths, negatives)

    def save(self, model_dir: str) -> None:
        """Saves the trained verification model."""
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
        """Returns the recognition algorithm used by this verifier."""
        raise NotImplementedError

    def needs_preprocessing(self) -> bool:
        """
        If True, samples are preprocessed before being fed to the recognition algorithm.
        See preprocess.py for further info.
        """
        return True

    def uses_negatives(self) -> bool:
        """If True, negative samples are passed to the _train() method."""
        return False

    def _predict(self, image: np.array) -> float:
        """
        Subclasses should override this method by returning
        the confidence associated with the specified image.
        """
        raise NotImplementedError

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        """Subclasses should override this by providing suitable model training logic."""
        raise NotImplementedError

    def _load(self, model_path: str) -> None:
        """Subclasses should override this by providing model loading logic."""
        raise NotImplementedError

    def _save(self, model_path: str) -> None:
        """Subclasses should override this by providing model saving logic."""
        raise NotImplementedError

    # Private

    def __learn_threshold(self, truths: List[np.array], negatives: Iterable[np.array]) -> None:
        """Learns a suitable threshold value."""
        truth_confidences = [c for c in (self._predict(i) for i in truths) if np.isfinite(c)]
        min_positive_confidence = min(truth_confidences)
        avg_positive_confidence = np.mean(truth_confidences)

        positive_threshold = (min_positive_confidence + avg_positive_confidence) / 2
        negative_threshold = min(self._predict(i) for i in negatives)

        if np.isfinite(negative_threshold):
            self.threshold = (negative_threshold + positive_threshold) / 2
        else:
            self.threshold = (max(truth_confidences) + positive_threshold) / 2

        if config.DEBUG:
            print('Minimum positive confidence: {:.2f}'.format(min_positive_confidence))
            print('Average positive confidence: {:.2f}'.format(avg_positive_confidence))
            print('Minimum negative confidence: {:.2f}'.format(negative_threshold))
            print('Learned threshold: {:.2f}'.format(self.threshold))


class OpenCVVerifier(FaceVerifier):
    """
    Wrapper class for verifiers using face recognition algorithms from OpenCV.
    The confidence value is returned by the OpenCV FaceRecognizer API.
    """

    # Public

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'OpenCVVerifier':
        if algo == RecognitionAlgo.EIGEN:
            return OpenCVVerifier(algo, cv2.face.EigenFaceRecognizer_create)
        elif algo == RecognitionAlgo.FISHER:
            return OpenCVVerifier(algo, cv2.face.FisherFaceRecognizer_create)
        else:
            return OpenCVVerifier(algo, cv2.face.LBPHFaceRecognizer_create)

    def __init__(self, algo: RecognitionAlgo, cv_rec_create):
        super().__init__()
        self.__algo = algo
        self.__create_rec = cv_rec_create
        self.__rec: cv2.face_BasicFaceRecognizer = None

    # Overrides

    def algo(self) -> RecognitionAlgo:
        return self.__algo

    def uses_negatives(self) -> bool:
        return self.algo() == RecognitionAlgo.FISHER

    def _predict(self, image: np.array) -> float:
        label, confidence = self.__rec.predict(image)
        return confidence if label == 0 else float('inf')

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:

        positives = np.asarray(list(positive_samples), dtype=np.int)
        negatives = np.asarray(list(islice(negative_samples, len(positives))), dtype=np.int)
        labels = ([0] * len(positives)) + ([1] * len(negatives))

        samples = np.concatenate((positives, negatives)) if len(negatives) > 0 else positives
        labels = np.asarray(labels, dtype=np.int)

        self.__rec = self.__create_rec()
        self.__rec.train(samples, labels)
        self.__print_debug_info(np.shape(positives[0]))

    def _load(self, model_path: str) -> None:
        self.__rec = self.__create_rec()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)

    # Private

    def __print_debug_info(self, sample_shape: np.array) -> None:
        algo = self.algo()

        if config.DEBUG and algo in [RecognitionAlgo.EIGEN, RecognitionAlgo.FISHER]:
            color_map = cv2.COLORMAP_JET if algo == RecognitionAlgo.EIGEN else cv2.COLORMAP_BONE
            eigenvectors = self.__rec.getEigenVectors()

            for i in range(np.shape(eigenvectors)[1]):
                image = img.normalized(np.reshape(eigenvectors[:, i], sample_shape))
                image = cv2.applyColorMap(image, color_map)
                img.save(image, config.Paths.DEBUG_DIR + '/ev{}.png'.format(i))


class FeaturesVerifier(FaceVerifier):
    """
    Performs verification by using a feature extractor.
    The confidence value represents the minimum distance between the feature vector extracted
    from the sample to verify and any of the feature vectors extracted from the training samples.
    """

    # Public

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'FeaturesVerifier':
        if algo == RecognitionAlgo.CNN:
            return FeaturesVerifier(algo, CNNFeatureExtractor())
        else:
            return FeaturesVerifier(algo, GeometricFeatureExtractor())

    def __init__(self, algo: RecognitionAlgo, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.__algo = algo
        self.__extractor = extractor
        self.__embeddings = []

    # Overrides

    def algo(self) -> RecognitionAlgo:
        return self.__algo

    def needs_preprocessing(self) -> bool:
        return False

    def _predict(self, image: np.array) -> float:
        sample = self.__extractor.extract_features(image)

        if sample is None:
            return float('inf')

        return min(np.linalg.norm(x - sample) for x in self.__embeddings)

    def _train(self, positive_samples: Iterable[np.array],
               negative_samples: Iterable[np.array]) -> None:
        del negative_samples  # Unused
        embeddings = (self.__extractor.extract_features(i) for i in positive_samples)
        self.__embeddings = [e for e in embeddings if e is not None]

    def _load(self, model_path: str) -> None:
        json_array = fileutils.load_json(model_path)
        if isinstance(json_array, list):
            self.__embeddings = [np.asarray(e) for e in json_array if isinstance(e, list)]

    def _save(self, model_path: str) -> None:
        embeddings = [e.tolist() for e in self.__embeddings]
        fileutils.save_json(embeddings, model_path)
