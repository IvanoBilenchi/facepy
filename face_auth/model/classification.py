import cv2.cv2 as cv2
import numpy as np
from os import path
from typing import Any, Dict, List, Optional

from face_auth import config
from . import fileutils, img
from .detector import FaceSample, StaticFaceDetector
from .geometry import Size
from .process import Pipeline, Step
from .recognition_algo import RecognitionAlgo


class FaceClassifier:

    class ConfigKey:
        ALGO = 'algo'
        LABELS = 'labels'

    class FileName:
        MODEL = 'model.dat'
        CONFIG = 'model_config.json'

    @classmethod
    def create(cls, algo: RecognitionAlgo) -> 'FaceClassifier':
        if algo in [RecognitionAlgo.EIGEN, RecognitionAlgo.FISHER, RecognitionAlgo.LBPH]:
            return OpenCVClassifier(algo)
        elif algo == RecognitionAlgo.EUCLIDEAN:
            # TODO: implement
            return OpenCVClassifier(RecognitionAlgo.EIGEN)
        else:
            # TODO: implement
            return OpenCVClassifier(RecognitionAlgo.EIGEN)

    @classmethod
    def from_dir(cls, dir_path: str) -> 'FaceClassifier':
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
        return self._labels

    def __init__(self) -> None:
        self._detector = StaticFaceDetector(scale_factor=1)
        self._labels: List[str] = None

    def predict(self, image: np.array) -> Optional[str]:
        face = self._detector.detect_main_face(image)
        return self.predict_sample(FaceSample(image, face)) if face else None

    def predict_sample(self, sample: FaceSample) -> str:
        image = self.__extract_face(sample, config.DEBUG)
        return self._predict(image)

    def train(self, data: Dict[str, List[FaceSample]]) -> None:
        processed_data: List[List[np.array]] = []
        self._labels = []

        for name, samples in data.items():
            samples = [self.__extract_face(s, config.DEBUG) for s in samples]
            processed_data.append(samples)
            self._labels.append(name)

        self._train(processed_data)

    def save(self, model_dir: str) -> None:
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
        raise NotImplementedError

    def needs_preprocessing(self) -> bool:
        return True

    def _predict(self, image: np.array) -> str:
        raise NotImplementedError

    def _train(self, data: List[List[np.array]]) -> None:
        raise NotImplementedError

    def _load(self, model_path: str) -> None:
        raise NotImplementedError

    def _save(self, model_path: str) -> None:
        raise NotImplementedError

    # Private

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


class OpenCVClassifier(FaceClassifier):

    def __init__(self, algo: RecognitionAlgo) -> None:
        super().__init__()
        self.__algo = algo

        if algo == RecognitionAlgo.EIGEN:
            self.__create_rec = cv2.face.EigenFaceRecognizer_create
        elif algo == RecognitionAlgo.FISHER:
            self.__create_rec = cv2.face.FisherFaceRecognizer_create
        else:
            self.__create_rec = cv2.face.LBPHFaceRecognizer_create

        self.__rec: cv2.face_BasicFaceRecognizer = None

    # Overrides

    def algo(self) -> RecognitionAlgo:
        return self.__algo

    def _predict(self, image: np.array) -> str:
        label, _ = self.__rec.predict(image)
        return self._labels[label]

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


# Utils


def _labels_for_data(data: List[List[Any]]) -> List[int]:
    labels_array = []

    for idx, sub_list in enumerate(data):
        labels_array.extend([idx] * len(sub_list))

    return labels_array


def _flatten_data(data: List[List[np.array]]) -> List[np.array]:
    return [image for images in data for image in images]
