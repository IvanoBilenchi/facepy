from enum import Enum
from typing import List, Optional

import cv2.cv2 as cv2
import dlib
import numpy as np

from facepy import config
from facepy.config import Paths
from .geometry import Face, Landmarks, Rect, Size


# Public


class RawModel:
    """Raw Dlib and OpenCV detection models."""

    _haar_detector = None
    _hog_detector = None
    _cnn_detector = None
    _shape_predictor = None
    _shape_predictor_small = None

    @classmethod
    def haar_detector(cls) -> cv2.CascadeClassifier:
        """OpenCV's Haar cascade detector."""
        if cls._haar_detector is None:
            cls._haar_detector = cv2.CascadeClassifier(Paths.HAAR_FACE_DETECTOR_MODEL)
        return cls._haar_detector

    @classmethod
    def hog_detector(cls) -> dlib.fhog_object_detector:
        """Dlib's HOG detector."""
        if cls._hog_detector is None:
            cls._hog_detector = dlib.get_frontal_face_detector()
        return cls._hog_detector

    @classmethod
    def cnn_detector(cls) -> dlib.cnn_face_detection_model_v1:
        """Dlib's CNN detector."""
        if cls._cnn_detector is None:
            cls._cnn_detector = dlib.cnn_face_detection_model_v1(Paths.CNN_FACE_DETECTOR_MODEL)
        return cls._cnn_detector

    @classmethod
    def shape_predictor(cls) -> dlib.shape_predictor:
        """Dlib's 68-point pose estimator."""
        if cls._shape_predictor is None:
            cls._shape_predictor = dlib.shape_predictor(Paths.FACE_LANDMARKS_MODEL)
        return cls._shape_predictor

    @classmethod
    def shape_predictor_small(cls) -> dlib.shape_predictor:
        """Dlib's 5-point pose estimator."""
        if cls._shape_predictor_small is None:
            cls._shape_predictor_small = dlib.shape_predictor(Paths.FACE_LANDMARKS_SMALL_MODEL)
        return cls._shape_predictor_small


class FaceSample:
    """Encapsulates a face image with geometry data."""

    def __init__(self, image: np.array, face: Face) -> None:
        self.image = image.copy()
        self.face = face

    def pose_is_frontal(self) -> bool:
        """Checks if landmarks correspond to a frontal pose."""
        return self.face.landmarks.pose_is_frontal()


class FaceDetector:
    """Abstract face detector base class."""

    class Algo(Enum):
        """Type of detection algorithm."""
        HAAR = 0
        HOG = 1
        CNN = 2

    # Public methods

    def get_main_face_sample(self, frame: np.array) -> Optional[FaceSample]:
        """Convenience method: runs the detector and returns a FaceSample instance."""
        face = self.detect_main_face(frame)
        return FaceSample(frame, face) if face else None

    # Must override

    def detect_main_face(self, frame: np.array) -> Optional[Face]:
        """
        Abstract method: returns the "main" face in the image. In the current implementation,
        the "main" face is the one enclosed in the rect nearest to the center of the image.
        """
        raise NotImplementedError


class StaticFaceDetector(FaceDetector):
    """
    Detects faces and landmarks in images.
    FaceDetector's abstract methods are implemented by wrapping existing OpenCV and Dlib detectors.
    """

    def __init__(self, algo: FaceDetector.Algo = FaceDetector.Algo[config.Detector.ALGORITHM],
                 scale_factor: int = config.Detector.SCALE_FACTOR) -> None:

        if scale_factor < 1:
            scale_factor = config.Detector.SCALE_FACTOR

        self.__algo = algo
        self.__scale_factor = scale_factor

    def detect_main_face(self, frame: np.array) -> Optional[Face]:
        faces = self.__detect_faces(frame)
        main_face_rect = Rect.nearest_to_center(faces, Size.of_image(frame).center)

        if main_face_rect is None:
            return None

        return Face(main_face_rect, self.__detect_landmarks(frame, main_face_rect))

    # Private

    def __detect_faces(self, frame: np.array) -> List[Rect]:
        """
        Detects faces by wrapping existing OpenCV and Dlib detectors.
        If 'scale_factor' is specified, the image is first resized to 100.0/scale_factor %
        in order to reduce detection latency at the cost of marginally reduced accuracy.
        """
        if self.__algo == FaceDetector.Algo.HAAR:
            detect_func = _haar_detect_faces
        elif self.__algo == FaceDetector.Algo.HOG:
            detect_func = _hog_detect_faces
        elif self.__algo == FaceDetector.Algo.CNN:
            detect_func = _cnn_detect_faces
        else:
            return []

        scale_factor = self.__scale_factor

        if scale_factor <= 1:
            return detect_func(frame)
        else:
            frame = cv2.resize(frame, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor)
            return [face.scaled(scale_factor) for face in detect_func(frame)]

    def __detect_landmarks(self, frame: np.array, rect: Rect) -> Landmarks:
        """Detects landmarks by wrapping Dlib's pose estimator."""
        landmarks = RawModel.shape_predictor()(frame, rect.to_dlib_rect())
        return Landmarks.from_dlib_landmarks(landmarks)


class VideoFaceDetector(FaceDetector):
    """
    Detects faces and landmarks in video frames.
    This class accounts for past detection history by exponentially averaging face bounding boxes
    and landmark points. Furthermore, detections skipped due to bad pose are filtered out
    up to a certain degree by accounting for detection success history.
    """

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

    def detect_main_face(self, frame: np.array) -> Optional[Face]:

        # Perform detection on single frame.
        face = self.__detector.detect_main_face(frame)

        # Filter invalid poses.
        if face and not face.landmarks.pose_is_valid():
            face = None

        # Approximate invalid poses with last detection.
        if face:
            self.__update_success_history(True)
        else:
            self.__update_success_history(False)
            face = self.__last_detection_cache()

            if face is None:
                return None

        # Fail if success rate is too low.
        if self.__compute_success_rate() < 0.6:
            return None

        # Approximate detector output and update last detection.
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


def _haar_detect_faces(frame: np.array) -> List[Rect]:
    """Wraps OpenCV's Haar cascades detector."""
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = RawModel.haar_detector()
    return [Rect(*f) for f in detector.detectMultiScale(temp_frame, 1.2, 5)]


def _hog_detect_faces(frame: np.array) -> List[Rect]:
    """Wraps Dlib's HOG detector."""
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = RawModel.hog_detector()
    return [Rect.from_dlib_rect(rect) for rect in detector(temp_frame)]


def _cnn_detect_faces(frame: np.array) -> List[Rect]:
    """Wraps Dlib's CNN detector."""
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = RawModel.cnn_detector()
    return [Rect.from_dlib_rect(face.rect) for face in detector(temp_frame)]
