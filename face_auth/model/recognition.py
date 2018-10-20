import cv2.cv2 as cv2
import numpy as np

from . import img
from .geometry import Landmarks, Size
from .process import Pipeline, Step
from face_auth import config


class FaceRecognitionModel:

    def __init__(self) -> None:
        self.__samples = []

    def add_sample(self, image: np.array, landmarks: Landmarks) -> None:
        self.__samples.append(FaceRecognizer.extract_face(image, landmarks))

    def remove_last_sample(self) -> None:
        self.__samples.pop()

    def sample_count(self) -> int:
        return len(self.__samples)

    def train(self, model_path: str) -> None:
        rec: cv2.face_LBPHFaceRecognizer = cv2.face.LBPHFaceRecognizer_create()
        rec.train(np.asarray(self.__samples), np.zeros(self.sample_count(), dtype=np.int))
        rec.save(model_path)


class FaceRecognizer:

    def __init__(self, model_path: str) -> None:
        self.__rec: cv2.face_LBPHFaceRecognizer = cv2.face.LBPHFaceRecognizer_create()
        self.__rec.read(model_path)

    def confidence_of_prediction(self, image: np.array, landmarks: Landmarks) -> float:
        label, confidence = self.__rec.predict(FaceRecognizer.extract_face(image, landmarks))
        return confidence if label == 0 else float('inf')

    def predict(self, image: np.array, landmarks: Landmarks) -> bool:
        return self.confidence_of_prediction(image, landmarks) < 30.0

    @classmethod
    def extract_face(cls, image: np.array, landmarks: Landmarks) -> np.array:
        rect = landmarks.square()
        shape = landmarks.outer_shape
        matrix = landmarks.alignment_matrix()

        return Pipeline.execute('Face extraction', image, config.DEBUG, [
            Step('To grayscale', img.to_grayscale),
            Step('Mask', lambda f: img.masked_to_shape(f, shape)),
            Step('Align', lambda f: img.transform(f, matrix, rect.size)),
            Step('Resize', lambda f: img.resized(f, Size(200, 200))),
            Step('Denoise', img.denoised),
            Step('Equalize', img.equalized)
        ])
