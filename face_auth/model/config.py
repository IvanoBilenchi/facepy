import cv2.data as cvdata
import sys
from os import path


class Paths:
    """Paths config namespace."""
    DIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
    RES_DIR = path.join(DIR, 'res')

    HAAR_FACE_DETECTOR_MODEL = path.join(cvdata.haarcascades, 'haarcascade_frontalface_default.xml')
    CNN_FACE_DETECTOR_MODEL = path.join(RES_DIR, 'mmod_human_face_detector.dat')
    FACE_LANDMARKS_MODEL = path.join(RES_DIR, 'shape_predictor_68_face_landmarks.dat')
