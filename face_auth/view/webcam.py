import cv2.cv2 as cv2
import numpy as np
from typing import Callable

from face_auth.model import img


# Constants

WINDOW_NAME = 'Webcam'


# Public functions

def start_capture(delegate: Callable[[np.array, int], None]):

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 100, 80)

    cam = cv2.VideoCapture(0)

    while True:
        key = cv2.waitKey(1)
        success, frame = cam.read()

        if success:
            frame = cv2.flip(frame, 1)
            frame = img.cropped_to_square(frame)
            delegate(frame, key)
            cv2.imshow(WINDOW_NAME, frame)

        if key == 27:
            break

    cv2.destroyAllWindows()
