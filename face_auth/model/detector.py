import cv2.cv2 as cv2
import dlib
import numpy as np

from enum import Enum
from typing import Callable, List

from face_auth import config
from .geometry import Landmarks, Rect


# Globals


__haar_detector = cv2.CascadeClassifier(config.Paths.HAAR_FACE_DETECTOR_MODEL)
__hog_detector = dlib.get_frontal_face_detector()
__cnn_detector = dlib.cnn_face_detection_model_v1(config.Paths.CNN_FACE_DETECTOR_MODEL)
__shape_predictor = dlib.shape_predictor(config.Paths.FACE_LANDMARKS_MODEL)


# Types


class DetectionAlgo(Enum):
    HAAR = 1
    HOG = 2
    CNN = 3

    @classmethod
    def default(cls) -> 'DetectionAlgo':
        return DetectionAlgo[config.Detector.ALGORITHM]


# Public functions


def detect_faces(frame: np.array, algo: DetectionAlgo = DetectionAlgo.default()) -> List[Rect]:
    if algo == DetectionAlgo.HAAR:
        return __detect_faces(frame, __haar_detect_faces)
    elif algo == DetectionAlgo.HOG:
        return __detect_faces(frame, __hog_detect_faces)
    elif algo == DetectionAlgo.CNN:
        return __detect_faces(frame, __cnn_detect_faces)
    else:
        return []


def detect_landmarks(frame: np.array, rect: Rect) -> Landmarks:
    landmarks = __shape_predictor(frame, rect.to_dlib_rect())
    return Landmarks.from_dlib_landmarks(landmarks)


# Private functions


def __detect_faces(frame: np.array, func: Callable[[np.array], List[Rect]]) -> List[Rect]:
    scale_factor = config.Detector.SCALE_FACTOR
    temp_frame = cv2.resize(frame, (0, 0), fx=1.0/scale_factor, fy=1.0/scale_factor)
    return [face.scaled(scale_factor) for face in func(temp_frame)]


def __haar_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [Rect(*f) for f in __haar_detector.detectMultiScale(temp_frame, 1.2, 5)]


def __hog_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [Rect.from_dlib_rect(rect) for rect in __hog_detector(temp_frame)]


def __cnn_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [Rect.from_dlib_rect(face.rect) for face in __cnn_detector(temp_frame)]
