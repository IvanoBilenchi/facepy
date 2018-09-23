from dlib import rectangle
from typing import NamedTuple


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
