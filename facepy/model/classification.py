from os import path
from typing import Any, Dict, List, Optional

import cv2.cv2 as cv2
import numpy as np

from facepy import config
from . import fileutils, preprocess
from .detector import FaceSample, StaticFaceDetector
from .feature_extractor import CNNFeatureExtractor, FeatureExtractor, GeometricFeatureExtractor
from .recognition_algo import RecognitionAlgo


class FaceClassifier:
    """
    Abstract face classifier class.

    Face classifiers are used to identify people depicted in images.
    """

    class ConfigKey:
        """Model configuration file keys."""
        ALGO = 'algo'
        LABELS = 'labels'

    class FileName:
        """Model file names."""
        MODEL = 'model.dat'
        CONFIG = 'model_config.json'

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'FaceClassifier':
        """Factory method: creates a face classifier using the specified algorithm."""
        if algo in [RecognitionAlgo.EIGEN, RecognitionAlgo.FISHER, RecognitionAlgo.LBPH]:
            return OpenCVClassifier.create(algo)
        else:
            return FeaturesClassifier.create(algo)

    @classmethod
    def from_dir(cls, dir_path: str) -> 'FaceClassifier':
        """Factory method: loads a pre-trained face verifier from the specified directory."""
        model_path = path.join(dir_path, cls.FileName.MODEL)
        config_path = path.join(dir_path, cls.FileName.CONFIG)

        cfg = fileutils.load_json(config_path)
        algo = RecognitionAlgo[cfg[cls.ConfigKey.ALGO]]

        classifier = cls.create(algo)
        classifier._labels = cfg[cls.ConfigKey.LABELS]
        classifier._load(model_path)

        return classifier

    @property
    def labels(self) -> List[str]:
        """Names of the people this classifier can identify."""
        return self._labels

    def __init__(self) -> None:
        self._detector = StaticFaceDetector(scale_factor=1)
        self._labels: List[str] = None

    def predict(self, image: np.array) -> Optional[str]:
        """Predicts the name of the person depicted in the specified image."""
        face = self._detector.detect_main_face(image)
        return self.predict_sample(FaceSample(image, face)) if face else None

    def predict_sample(self, sample: FaceSample) -> str:
        """Predicts the name of the person depicted in the specified sample."""
        image = preprocess.prepare_for_recognition(sample,
                                                   preprocess=self.needs_preprocessing(),
                                                   debug=config.DEBUG)
        return self._labels[self._predict(image)]

    def train(self, data: Dict[str, List[FaceSample]]) -> None:
        """Trains the classifier. 'data' is a dictionary mapping labels to training samples."""
        processed_data: List[List[np.array]] = []
        self._labels = []

        for name, samples in data.items():
            samples = [
                preprocess.prepare_for_recognition(sample,
                                                   preprocess=self.needs_preprocessing(),
                                                   debug=config.DEBUG)
                for sample in samples
            ]

            processed_data.append(samples)
            self._labels.append(name)

        self._train(processed_data)

    def save(self, model_dir: str) -> None:
        """Saves the trained classification model."""
        fileutils.create_dir(model_dir)
        model_path = path.join(model_dir, self.FileName.MODEL)
        config_path = path.join(model_dir, self.FileName.CONFIG)

        self._save(model_path)

        cfg = {
            self.ConfigKey.ALGO: self.algo().name,
            self.ConfigKey.LABELS: self._labels
        }

        fileutils.create_parent_dir(config_path)
        fileutils.save_json(cfg, config_path)

    # Must override

    def algo(self) -> RecognitionAlgo:
        """Returns the recognition algorithm used by this classifier."""
        raise NotImplementedError

    def needs_preprocessing(self) -> bool:
        """
        If True, samples are preprocessed before being fed to the recognition algorithm.
        See preprocess.py for further info.
        """
        return True

    def _predict(self, image: np.array) -> int:
        """
        Subclasses should override this method by returning the index corresponding to the
        predicted label for the specified image.
        """
        raise NotImplementedError

    def _train(self, data: List[List[np.array]]) -> None:
        """Subclasses should override this by providing suitable model training logic."""
        raise NotImplementedError

    def _load(self, model_path: str) -> None:
        """Subclasses should override this by providing model loading logic."""
        raise NotImplementedError

    def _save(self, model_path: str) -> None:
        """Subclasses should override this by providing model saving logic."""
        raise NotImplementedError


class OpenCVClassifier(FaceClassifier):
    """Wrapper class for classifiers using face recognition algorithms from OpenCV."""

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'OpenCVClassifier':
        if algo == RecognitionAlgo.EIGEN:
            return OpenCVClassifier(algo, cv2.face.EigenFaceRecognizer_create)
        elif algo == RecognitionAlgo.FISHER:
            return OpenCVClassifier(algo, cv2.face.FisherFaceRecognizer_create)
        else:
            return OpenCVClassifier(algo, cv2.face.LBPHFaceRecognizer_create)

    def __init__(self, algo: RecognitionAlgo, cv_rec_create) -> None:
        super().__init__()
        self.__algo = algo
        self.__create_rec = cv_rec_create
        self.__rec: cv2.face_BasicFaceRecognizer = None

    # Overrides

    def algo(self) -> RecognitionAlgo:
        return self.__algo

    def _predict(self, image: np.array) -> int:
        return self.__rec.predict(image)[0]

    def _train(self, data: List[List[np.array]]) -> None:
        img_array = np.asarray(_flatten_data(data))
        labels_array = np.asarray(_labels_for_data(data))
        self.__rec = self.__create_rec()
        self.__rec.train(img_array, labels_array)

    def _load(self, model_path: str) -> None:
        self.__rec = self.__create_rec()
        self.__rec.read(model_path)

    def _save(self, model_path: str) -> None:
        self.__rec.save(model_path)


class FeaturesClassifier(FaceClassifier):
    """1-NN classifier using features extracted via a FeatureExtractor object."""

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'FeaturesClassifier':
        if algo == RecognitionAlgo.CNN:
            return FeaturesClassifier(algo, CNNFeatureExtractor())
        else:
            return FeaturesClassifier(algo, GeometricFeatureExtractor())

    def __init__(self, algo: RecognitionAlgo, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.__algo = algo
        self.__extractor = extractor
        self.__model: List[List[np.array]] = []

    # Overrides

    def algo(self) -> RecognitionAlgo:
        return self.__algo

    def needs_preprocessing(self) -> bool:
        return False

    def _predict(self, image: np.array) -> int:
        embedding = self.__extractor.extract_features(image)

        label = min((min(np.linalg.norm(x - embedding) for x in e), i)
                    for i, e in enumerate(self.__model))[1]

        return label

    def _train(self, data: List[List[np.array]]) -> None:
        self.__model = [[self.__extractor.extract_features(x) for x in s] for s in data]

    def _load(self, model_path: str) -> None:
        json_array = fileutils.load_json(model_path)
        if isinstance(json_array, list):
            self.__model = [
                [np.asarray(e) for e in x if isinstance(e, list)]
                for x in json_array if isinstance(x, list)
            ]

    def _save(self, model_path: str) -> None:
        model = [[x.tolist() for x in y] for y in self.__model]
        fileutils.save_json(model, model_path)


# Utils


def _labels_for_data(data: List[List[Any]]) -> List[int]:
    labels_array = []

    for idx, sub_list in enumerate(data):
        labels_array.extend([idx] * len(sub_list))

    return labels_array


def _flatten_data(data: List[List[np.array]]) -> List[np.array]:
    return [image for images in data for image in images]
