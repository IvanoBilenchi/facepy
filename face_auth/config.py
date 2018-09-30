import cv2.cv2 as cv2
import cv2.data as cvdata
import sys
from os import path


class Paths:
    """Paths config namespace."""
    DIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
    RES_DIR = path.join(DIR, 'res')
    USER_DIR = path.join(DIR, 'user')

    HAAR_FACE_DETECTOR_MODEL = path.join(cvdata.haarcascades, 'haarcascade_frontalface_default.xml')
    CNN_FACE_DETECTOR_MODEL = path.join(RES_DIR, 'mmod_human_face_detector.dat')
    FACE_LANDMARKS_MODEL = path.join(RES_DIR, 'shape_predictor_68_face_landmarks.dat')

    FACE_IMAGE = path.join(USER_DIR, 'face.png')


class Detector:
    """Detector config namespace."""
    SCALE_FACTOR = 5
    ALGORITHM = 'HOG'


class Renderer:
    """Renderer config namespace."""
    WINDOW_NAME = 'Webcam'
    VIDEO_SIZE = (450, 450)

    class FPS:
        """FPS renderer config namespace."""
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_HEIGHT = 14
        FONT_COLOR = (0, 255, 0)
        FONT_THICKNESS = 1
        FONT_SCALE = cv2.getFontScaleFromHeight(FONT, FONT_HEIGHT)
        LINE_TYPE = cv2.LINE_AA

    class Rect:
        """Rect renderer config namespace."""
        THICKNESS = 3
        COLOR = (255, 0, 0)
        LINE_TYPE = cv2.LINE_AA

    class Landmarks:
        """Landmarks renderer config namespace."""
        THICKNESS = 2
        BASE_COLOR = (0, 255, 0)
        EYE_COLOR = (255, 255, 255)
        MOUTH_COLOR = (0, 0, 255)
        LINE_TYPE = cv2.LINE_AA