import numpy as np
from .geometry import Rect


def size(image: np.array) -> (int, int):
    shape = image.shape[:2]
    return shape[1], shape[0]


def cropped(image: np.array, rect: Rect) -> np.array:
    return image[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width]


def cropped_to_square(image: np.array) -> np.array:
    w, h = size(image)
    rect = Rect((w - h) // 2, 0, h, h) if w > h else Rect(0, (h - w) // 2, w, w)
    return cropped(image, rect)
