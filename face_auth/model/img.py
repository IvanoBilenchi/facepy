import cv2.cv2 as cv2
import numpy as np
import os
from contextlib import suppress
from .geometry import Rect, Size


def size(image: np.array) -> Size:
    shape = image.shape[:2]
    return Size(shape[1], shape[0])


def cropped(image: np.array, rect: Rect) -> np.array:
    return image[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width]


def cropped_to_square(image: np.array) -> np.array:
    w, h = size(image)
    rect = Rect((w - h) // 2, 0, h, h) if w > h else Rect(0, (h - w) // 2, w, w)
    return cropped(image, rect)


def save(image: np.array, path: str) -> bool:
    with suppress(FileNotFoundError):
        os.unlink(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    return True
