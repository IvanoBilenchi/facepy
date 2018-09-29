import numpy as np
from dlib import rectangle
from typing import List, NamedTuple


class Point(NamedTuple):
    """Models a point."""
    x: int = 0
    y: int = 0


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

    @classmethod
    def from_dlib_landmarks(cls, landmarks) -> 'Landmarks':
        points = [Point(part.x, part.y) for part in landmarks.parts()]
        return Landmarks(
            chin=points[0:17],
            left_eyebrow=points[17:22],
            right_eyebrow=points[22:27],
            left_eye=points[36:42],
            right_eye=points[42:48],
            nose_bridge=points[27:31],
            nose_tip=points[31:36],
            top_lip=points[48:55] + points[64:59:-1],
            bottom_lip=points[54:60] + [points[48], points[60]] + points[:63:-1]
        )

    @classmethod
    def to_numpy(cls, pts: List[Point]) -> np.array:
        return np.int32([np.asarray(pts)])

    @property
    def all(self) -> List[Point]:
        return (self.chin + self.left_eyebrow + self.right_eyebrow + self.left_eye +
                self.right_eye + self.nose_bridge + self.nose_tip + self.top_lip + self.bottom_lip)
