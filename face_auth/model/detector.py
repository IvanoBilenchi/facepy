import cv2.cv2 as cv2
import dlib
import numpy as np

from enum import Enum
from typing import Callable, List, Optional

from face_auth import config
from .geometry import Face, Landmarks, Size, Rect


# Public


class FaceDetector:

    class Algo(Enum):
        HAAR = 0
        HOG = 1
        CNN = 2

    def detect_faces(self, frame: np.array) -> List[Rect]:
        raise NotImplementedError

    def detect_landmarks(self, frame: np.array, rect: Rect) -> Landmarks:
        raise NotImplementedError

    def detect_main_face(self, frame: np.array) -> Optional[Face]:
        raise NotImplementedError


class StaticFaceDetector(FaceDetector):

    def __init__(self, algo: FaceDetector.Algo = FaceDetector.Algo[config.Detector.ALGORITHM],
                 scale_factor: int = config.Detector.SCALE_FACTOR) -> None:

        if scale_factor < 1:
            scale_factor = config.Detector.SCALE_FACTOR

        self.__algo = algo
        self.__scale_factor = scale_factor

    def detect_faces(self, frame: np.array) -> List[Rect]:
        if self.__algo == FaceDetector.Algo.HAAR:
            detect_func = _haar_detect_faces
        elif self.__algo == FaceDetector.Algo.HOG:
            detect_func = _hog_detect_faces
        elif self.__algo == FaceDetector.Algo.CNN:
            detect_func = _cnn_detect_faces
        else:
            return []

        return _detect_faces(frame, self.__scale_factor, detect_func)

    def detect_landmarks(self, frame: np.array, rect: Rect) -> Landmarks:
        landmarks = _shape_predictor(frame, rect.to_dlib_rect())
        return Landmarks.from_dlib_landmarks(landmarks)

    def detect_main_face(self, frame: np.array) -> Optional[Face]:
        faces = self.detect_faces(frame)
        main_face_rect = Rect.nearest_to_center(faces, Size.of_image(frame).center)

        if main_face_rect is None:
            return None

        return Face(main_face_rect, self.detect_landmarks(frame, main_face_rect))


class VideoFaceDetector(FaceDetector):

    def __init__(self, algo: FaceDetector.Algo = FaceDetector.Algo[config.Detector.ALGORITHM],
                 scale_factor: int = config.Detector.SCALE_FACTOR,
                 smoothness: float = config.Detector.SMOOTHNESS) -> None:

        if smoothness > 1.0 or smoothness < 0.0:
            smoothness = config.Detector.SMOOTHNESS

        self.__alpha = 1.0 - smoothness
        self.__detector = StaticFaceDetector(algo, scale_factor)

        self.__last_detection: Face = None
        self.__last_detection_approximation_count = 0
        self.__detection_success_history = []

    def detect_faces(self, frame: np.array) -> List[Rect]:
        return self.__detector.detect_faces(frame)

    def detect_landmarks(self, frame: np.array, rect: Rect) -> Landmarks:
        return self.__detector.detect_landmarks(frame, rect)

    def detect_main_face(self, frame: np.array) -> Optional[Face]:
        face = self.__detector.detect_main_face(frame)

        if face and not face.landmarks.pose_is_valid():
            face = None

        if face:
            self.__update_success_history(True)
        else:
            self.__update_success_history(False)
            face = self.__last_detection_cache()

            if face is None:
                return None

        if self.__compute_success_rate() < 0.6:
            return None

        if self.__last_detection:
            face = face.weighting_previous(self.__last_detection, self.__alpha)

        self.__last_detection = face

        return face

    # Private

    def __update_success_history(self, success: bool) -> None:
        if len(self.__detection_success_history) == 30:
            self.__detection_success_history.pop(0)
        self.__detection_success_history.append(1.0 if success else 0.0)

    def __compute_success_rate(self) -> float:
        count = len(self.__detection_success_history)
        return sum(self.__detection_success_history) / count

    def __last_detection_cache(self) -> Optional[Face]:
        face = None

        if self.__last_detection is not None:
            if self.__last_detection_approximation_count < 10:
                face = self.__last_detection
                self.__last_detection_approximation_count += 1
            else:
                self.__last_detection = None
                self.__last_detection_approximation_count = 0

        return face


# Private


_haar_detector = cv2.CascadeClassifier(config.Paths.HAAR_FACE_DETECTOR_MODEL)
_hog_detector = dlib.get_frontal_face_detector()
_cnn_detector = dlib.cnn_face_detection_model_v1(config.Paths.CNN_FACE_DETECTOR_MODEL)
_shape_predictor = dlib.shape_predictor(config.Paths.FACE_LANDMARKS_MODEL)


def _detect_faces(frame: np.array, scale_factor: int,
                  func: Callable[[np.array], List[Rect]]) -> List[Rect]:
    if scale_factor <= 1:
        return func(frame)
    else:
        frame = cv2.resize(frame, (0, 0), fx=1.0/scale_factor, fy=1.0/scale_factor)
        return [face.scaled(scale_factor) for face in func(frame)]


def _haar_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [Rect(*f) for f in _haar_detector.detectMultiScale(temp_frame, 1.2, 5)]


def _hog_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [Rect.from_dlib_rect(rect) for rect in _hog_detector(temp_frame)]


def _cnn_detect_faces(frame: np.array) -> List[Rect]:
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [Rect.from_dlib_rect(face.rect) for face in _cnn_detector(temp_frame)]
