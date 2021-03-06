import cv2.cv2 as cv2
import cv2.data as cvdata
import sys
from os import path

from .view import color


DEBUG = False
WEBCAM = 0


class Paths:
    """Paths config namespace."""
    DIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
    RES_DIR = path.join(DIR, 'res')
    USER_DIR = path.join(DIR, 'user')
    DEBUG_DIR = path.join(USER_DIR, 'debug')

    HAAR_FACE_DETECTOR_MODEL = path.join(cvdata.haarcascades, 'haarcascade_frontalface_default.xml')
    CNN_FACE_DETECTOR_MODEL = path.join(RES_DIR, 'mmod_human_face_detector.dat')
    FACE_LANDMARKS_MODEL = path.join(RES_DIR, 'shape_predictor_68_face_landmarks.dat')
    FACE_LANDMARKS_SMALL_MODEL = path.join(RES_DIR, 'shape_predictor_5_face_landmarks.dat')
    CNN_FACE_DESCRIPTOR_MODEL = path.join(RES_DIR, 'dlib_face_recognition_resnet_model_v1.dat')

    DATASET_DIR = path.join(RES_DIR, 'lfw')
    VERIFICATION_MODEL_DIR = path.join(USER_DIR, 'verification')
    CLASSIFICATION_MODEL_DIR = path.join(USER_DIR, 'classification')


class Detector:
    """Detector config namespace."""
    ALGORITHM = 'HOG'
    SCALE_FACTOR = 5
    SMOOTHNESS = 0.55


class Recognizer:
    """Recognizer config namespace."""
    ALGORITHM = 'CNN'
    VERIFICATION_POSITIVE_TRAINING_SAMPLES = 10
    VERIFICATION_POSITIVE_TRAINING_SAMPLES_SPLIT = 0.2
    VERIFICATION_NEGATIVE_TRAINING_SAMPLES = 50
    CLASSIFICATION_MIN_SAMPLES = 50
    CLASSIFICATION_TRAINING_SAMPLES = 25


class Renderer:
    """Renderer config namespace."""
    WINDOW_NAME = 'Webcam'
    VIDEO_SIZE = (450, 450)

    class Label:
        """Label renderer config namespace."""
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_HEIGHT = 14
        FONT_COLOR = color.WHITE
        FONT_THICKNESS = 1
        FONT_SCALE = cv2.getFontScaleFromHeight(FONT, FONT_HEIGHT)
        LINE_TYPE = cv2.LINE_AA
        PADDING = 10
        BAR_ALPHA = 0.6

    class Landmarks:
        """Landmarks renderer config namespace."""
        THICKNESS = 2
        BASE_COLOR = color.GREEN
        EYE_COLOR = color.WHITE
        MOUTH_COLOR = color.RED
        ALPHA = 0.6
        LINE_TYPE = cv2.LINE_AA
