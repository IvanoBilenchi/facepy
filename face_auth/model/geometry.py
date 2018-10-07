import numpy as np
import sys
from dlib import rectangle
from typing import List, NamedTuple


class Point(NamedTuple):
    """Models a point."""
    x: int = 0
    y: int = 0

    @classmethod
    def to_numpy(cls, pts: List['Point']) -> np.array:
        return np.int32([np.asarray(pts)])

    @classmethod
    def with_new_origin(cls, pts: List['Point'], origin: 'Point') -> List['Point']:
        ox = origin.x
        oy = origin.y
        return [Point(p.x - ox, p.y - oy) for p in pts]


class Size(NamedTuple):
    """Models sizes."""
    width: int = 0
    height: int = 0


class Rect(NamedTuple):
    """Models a rectangle."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @classmethod
    def with_size(cls, size: Size) -> 'Rect':
        return Rect(0, 0, *size)

    @classmethod
    def from_dlib_rect(cls, rect: rectangle) -> 'Rect':
        return Rect(rect.left(), rect.top(), rect.width(), rect.height())

    @property
    def top_left(self) -> Point:
        return Point(self.x, self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Point:
        return Point(self.x + self.width // 2, self.y + self.height // 2)

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    def scaled(self, scale_factor: int) -> 'Rect':
        return Rect(self.x * scale_factor, self.y * scale_factor,
                    self.width * scale_factor, self.height * scale_factor)

    def shrunk_to_square(self) -> 'Rect':
        x, y, w, h = self.x, self.y, self.width, self.height

        if w == h:
            return self

        return Rect(x + (w - h) // 2, y, h, h) if w > h else Rect(x, y + (h - w) // 2, w, w)

    def expanded_to_square(self) -> 'Rect':
        x, y, w, h = self.x, self.y, self.width, self.height

        if w == h:
            return self

        return Rect(x, y - (w - h) // 2, w, w) if w > h else Rect(x - (h - w) // 2, y, h, h)

    def to_dlib_rect(self) -> rectangle:
        return rectangle(self.x, self.y, self.x + self.width, self.y + self.height)


class Landmarks(NamedTuple):
    """Models face landmarks."""
    chin: List[Point]
    left_eyebrow: List[Point]
    right_eyebrow: List[Point]
    left_eye: List[Point]
    right_eye: List[Point]
    nose_bridge: List[Point]
    nose_tip: List[Point]
    top_lip: List[Point]
    bottom_lip: List[Point]
    outer_shape: List[Point]

    @classmethod
    def from_dlib_landmarks(cls, landmarks) -> 'Landmarks':
        points = [Point(part.x, part.y) for part in landmarks.parts()]
        return Landmarks(
            chin=points[:17],
            left_eyebrow=points[17:22],
            right_eyebrow=points[22:27],
            left_eye=points[36:42],
            right_eye=points[42:48],
            nose_bridge=points[27:31],
            nose_tip=points[31:36],
            top_lip=points[48:55] + points[64:59:-1],
            bottom_lip=points[54:60] + [points[48], points[60]] + points[:63:-1],
            outer_shape=points[18:20] + points[24:26] + points[16::-1]
        )

    @property
    def all(self) -> List[Point]:
        return (self.chin + self.left_eyebrow + self.right_eyebrow + self.left_eye +
                self.right_eye + self.nose_bridge + self.nose_tip + self.top_lip + self.bottom_lip)

    @property
    def rect(self) -> Rect:
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0

        for point in self.left_eyebrow:
            if point.x < min_x:
                min_x = point.x

            if point.y < min_y:
                min_y = point.y

        for point in self.right_eyebrow:
            if point.x > max_x:
                max_x = point.x

            if point.y < min_y:
                min_y = point.y

        for point in self.chin:
            if point.x > max_x:
                max_x = point.x

            if point.x < min_x:
                min_x = point.x

            if point.y > max_y:
                max_y = point.y

        return Rect(min_x, min_y, max_x - min_x, max_y - min_y)
