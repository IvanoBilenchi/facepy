import cv2.cv2 as cv2
import numpy as np
from threading import Lock, Thread

from facepy import config


class WebcamStream:

    def __init__(self) -> None:
        self.__cam: cv2.VideoCapture = None
        self.__frame: np.array = None

        self.__input_thread = Thread(name='Webcam input',
                                     target=self.__acquire_loop,
                                     daemon=True)
        self.__running = False
        self.__frame_lock = Lock()

    def start(self) -> None:
        self.__cam = cv2.VideoCapture(config.WEBCAM)
        self.__running = True
        self.__input_thread.start()

    def stop(self) -> None:
        self.__running = False
        self.__cam.release()

    def get_frame(self) -> np.array:
        with self.__frame_lock:
            return self.__frame

    # Private

    def __acquire_loop(self) -> None:
        while self.__running:
            success, frame = self.__cam.read()
            if success:
                self.__set_frame(frame)

    def __set_frame(self, frame: np.array) -> np.array:
        with self.__frame_lock:
            self.__frame = frame
