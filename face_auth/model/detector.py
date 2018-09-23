import cv2.cv2 as cv2
import dlib
import numpy as np

from enum import Enum
from typing import Callable, List

from .config import Paths
from .geometry import Rect


# Globals

__haar_detector = cv2.CascadeClassifier(Paths.HAAR_FACE_DETECTOR_MODEL)
__hog_detector = dlib.get_frontal_face_detector()
__cnn_detector = dlib.cnn_face_detection_model_v1(Paths.CNN_FACE_DETECTOR_MODEL)


# Types

class DetectionAlgo(Enum):
    HAAR = 1
    HOG = 2
    CNN = 3


# Public functions

def detect_faces(frame: np.array, algo: DetectionAlgo = DetectionAlgo.HAAR) -> List[Rect]:
    if algo == DetectionAlgo.HAAR:
        return __detect_faces(frame, __haar_detect_faces)
    elif algo == DetectionAlgo.HOG:
        return __detect_faces(frame, __hog_detect_faces)
    elif algo == DetectionAlgo.CNN:
        return __detect_faces(frame, __cnn_detect_faces)
    else:
        return []


# Private functions

def __detect_faces(frame: np.array, func: Callable[[np.array], List[Rect]]) -> List[Rect]:
    scale_factor = 4
    temp_frame = cv2.resize(frame, (0, 0), fx=1.0/scale_factor, fy=1.0/scale_factor)
    return [face.scaled(scale_factor) for face in func(temp_frame)]


def __haar_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [Rect(*f) for f in __haar_detector.detectMultiScale(temp_frame, 1.2, 5)]


def __hog_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [Rect(f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top())
            for f in __hog_detector(temp_frame)]


def __cnn_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = []

    for face in __cnn_detector(temp_frame):
        f = face.rect
        faces.append(Rect(f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top()))

    return faces
