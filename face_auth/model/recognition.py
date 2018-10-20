import cv2.cv2 as cv2
import numpy as np
import sys
from typing import NamedTuple

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

    def __init__(self) -> None:
        self.threshold = 0.0
        self.__rec: cv2.face_LBPHFaceRecognizer = None

    def confidence_of_prediction(self, sample: FaceSample) -> float:
        label, confidence = self.__rec.predict(FaceRecognizer.extract_face(sample))
        confidence = confidence if label == 0 else float('inf')

        if config.DEBUG:
            print('Prediction confidence: {:.2f} (threshold: {:.2f})'.format(confidence,
                                                                             self.threshold))
        return confidence

    def predict(self, sample: FaceSample) -> bool:
        return self.confidence_of_prediction(sample) < self.threshold

    def train(self, ground_truth: FaceSample, samples: [FaceSample],
              detector: FaceDetector, dataset: Dataset) -> None:

        ground_truth = FaceRecognizer.extract_face(ground_truth)
        samples = [FaceRecognizer.extract_face(s, debug=False) for s in samples]

        self.__rec = cv2.face.LBPHFaceRecognizer_create()
        self.__rec.train(np.asarray(samples), np.zeros(len(samples), dtype=np.int))
        self.__learn_threshold(ground_truth, detector, dataset)

    def load(self, model_path: str, config_path: str) -> None:
        try:
            self.__rec = cv2.face.LBPHFaceRecognizer_create()
            self.__rec.read(model_path)

            with open(config_path, mode='r') as config_file:
                self.threshold = float(config_file.read().strip())
        except Exception:
            raise FileNotFoundError('Could not find a trained model.')

    def save(self, model_path: str, config_path: str) -> None:
        self.__rec.save(model_path)

        with open(config_path, mode='w') as config_file:
            config_file.write('{:.2f}'.format(self.threshold))

    @classmethod
    def extract_face(cls, sample: FaceSample, debug: bool = config.DEBUG) -> np.array:
        rect = sample.landmarks.square()
        shape = sample.landmarks.outer_shape
        matrix = sample.landmarks.alignment_matrix()

        return Pipeline.execute('Face extraction', sample.image, debug, [
            Step('To grayscale', img.to_grayscale),
            Step('Mask', lambda f: img.masked_to_shape(f, shape)),
            Step('Align', lambda f: img.transform(f, matrix, rect.size)),
            Step('Resize', lambda f: img.resized(f, Size(100, 100))),
            Step('Denoise', img.denoised),
            Step('Equalize', img.equalized)
        ])

    # Private

    def __learn_threshold(self, ground_truth: np.array,
                          detector: FaceDetector, dataset: Dataset) -> None:
        min_confidence = sys.maxsize
        ground_confidence = self.__rec.predict(ground_truth)[1]

        for image in dataset.training_samples():
            face = detector.detect_main_face(image)

            if face is None:
                continue

            confidence = self.confidence_of_prediction(FaceSample(image, face.landmarks))

            if confidence < min_confidence:
                min_confidence = confidence

        self.threshold = (min_confidence + ground_confidence) / 2

        if config.DEBUG:
            print('Ground confidence: {:.2f}'.format(ground_confidence))
            print('Best confidence in training set: {:.2f}'.format(min_confidence))
            print('Learned treshold: {:.2f}'.format(self.threshold))
