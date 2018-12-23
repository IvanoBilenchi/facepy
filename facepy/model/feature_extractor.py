import cv2.cv2 as cv2
import dlib
import numpy as np
import os
from typing import List, Optional

from . import img
from .detector import RawModel, StaticFaceDetector
from .geometry import Landmarks, Point
from facepy import config
from facepy.view import geometry_renderer


class FeatureExtractor:
    """Abstract feature extractor class."""

    # Must override

    def extract_features(self, image: np.array) -> Optional[np.array]:
        raise NotImplementedError


class GeometricFeatureExtractor(FeatureExtractor):

    # Public

    def __init__(self) -> None:
        self.__detector = StaticFaceDetector(scale_factor=1)

    def extract_features(self, image: np.array) -> Optional[np.array]:
        face = self.__detector.detect_main_face(image)

        if face is None:
            return None

        traits = self.__rec_traits(face.landmarks)

        if config.DEBUG:
            new_image = image.copy()
            geometry_renderer.draw_points(new_image, traits)
            img.save(new_image, os.path.join(config.Paths.DEBUG_DIR, 'features.png'))

        n = len(traits)
        norm_factor = face.landmarks.left_eye[0].distance(face.landmarks.right_eye[3])

        # Size of the feature vector is given by (n choose 2)
        embedding_size = n * (n - 1) // 2
        embedding = np.zeros(embedding_size)
        idx = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
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

    def __distance(self, embedding: np.array, other: np.array) -> float:
        return np.linalg.norm(embedding - other)


class CNNFeatureExtractor(FeatureExtractor):

    # Public

    def __init__(self) -> None:
        self.__net = dlib.face_recognition_model_v1(config.Paths.CNN_FACE_DESCRIPTOR_MODEL)

    def extract_features(self, image: np.array) -> Optional[np.array]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = RawModel.hog_detector()(image)
        embedding = None

        if detections is not None and len(detections) > 0:
            landmarks = RawModel.shape_predictor_small()(image, detections[0])
            embedding = np.array(self.__net.compute_face_descriptor(image, landmarks))

        return embedding
