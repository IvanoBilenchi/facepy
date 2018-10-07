import cv2.cv2 as cv2
import numpy as np
import os
from contextlib import suppress
from .geometry import Point, Rect, Size


def is_colored(image: np.array) -> bool:
    return len(image.shape) == 3


def size(image: np.array) -> Size:
    shape = image.shape[:2]
    return Size(shape[1], shape[0])


def resized(image: np.array, new_size: Size) -> np.array:
    return cv2.resize(image, new_size)


def cropped(image: np.array, rect: Rect) -> np.array:
    return image[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width]


def cropped_to_square(image: np.array) -> np.array:
    return cropped(image, Rect.with_size(size(image)).shrunk_to_square())


def flipped_horizontally(image: np.array) -> np.array:
    return cv2.flip(image, 1)


def masked_to_shape(image: np.array, shape: [Point]) -> np.array:
    mask = np.zeros(image.shape, dtype=np.uint8)
    channel_count = image.shape[2] if is_colored(image) else 1
    mask_color = (255,) * channel_count
    cv2.fillPoly(mask, Point.to_numpy(shape), mask_color)
    return cv2.bitwise_and(image, mask)


def to_grayscale(image: np.array) -> np.array:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoised(image: np.array) -> np.array:
    if is_colored(image):
        return cv2.fastNlMeansDenoisingColored(image)
    else:
        return cv2.fastNlMeansDenoising(image)


def save(image: np.array, path: str) -> bool:
    with suppress(FileNotFoundError):
        os.unlink(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    return True
