import cv2.cv2 as cv2
import math
import numpy as np
import sys
from dlib import rectangle
from typing import List, NamedTuple, Optional


class Point(NamedTuple):
    """Models a point."""
    x: int = 0
    y: int = 0

    @classmethod
    def to_numpy(cls, pts: List['Point']) -> np.array:
        return np.int32([np.asarray(pts)])

    @classmethod
    def from_numpy(cls, pts: np.array) -> List['Point']:
        return [Point(p[0], p[1]) for p in pts]

    @classmethod
    def with_new_origin(cls, pts: List['Point'], origin: 'Point') -> List['Point']:
        ox, oy = origin.x, origin.y
        return [Point(p.x - ox, p.y - oy) for p in pts]

    @classmethod
    def mean(cls, pts: List['Point']) -> 'Point':
        x, y, n = 0, 0, len(pts)

        if n == 0:
            return Point(0, 0)

        for point in pts:
            x += point.x
            y += point.y

        return Point(x // n, y // n)

    def distance(self, other: 'Point') -> int:
        return int(math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


class Size(NamedTuple):
    """Models sizes."""
    width: int = 0
    height: int = 0

    @classmethod
    def of_image(cls, image: np.array) -> 'Size':
        shape = image.shape[:2]
        return Size(shape[1], shape[0])

    @property
    def center(self) -> Point:
        return Point(self.width // 2, self.height // 2)


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

    @classmethod
    def nearest_to_center(cls, rects: List['Rect'], center: Point) -> Optional['Rect']:
        min_distance = sys.maxsize
        nearest = None

        for rect in rects:
            distance = rect.center.distance(center)

            if distance < min_distance:
                min_distance = distance
                nearest = rect

        return nearest

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

    def rect(self) -> Rect:
        min_x, min_y, max_x, max_y = sys.maxsize, sys.maxsize, 0, 0

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

    def square(self) -> Rect:
        return Rect.expanded_to_square(self.rect())

    def alignment_matrix(self) -> np.array:
        # Get the center of the eyes
        left_eye = Point.mean(self.left_eye)
        right_eye = Point.mean(self.right_eye)
        face_square = self.square()

        # Compute tilt
        delta_y = right_eye.y - left_eye.y
        delta_x = right_eye.x - left_eye.x
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Normalized eye positions
        out_left_eye_x, out_left_eye_y = 0.3, 0.20
        out_right_eye_x, out_right_eye_y = 1.0 - out_left_eye_x, 1.0 - out_left_eye_y

        # Compute scale of output image
        dist = np.sqrt((delta_x ** 2) + (delta_y ** 2))
        out_dist = (out_right_eye_x - out_left_eye_x) * face_square.width
        scale = out_dist / dist

        # Compute rotation center point
        eyes_center = Point.mean([left_eye, right_eye])

        # Compute rotation matrix
        matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update translation components
        matrix[0, 2] += (face_square.width * 0.5 - eyes_center.x)
        matrix[1, 2] += (face_square.height * out_left_eye_y - eyes_center.y)

        return matrix

    def apply_transform(self, matrix) -> 'Landmarks':
        out = []

        for landmark in [self.chin, self.left_eyebrow, self.right_eyebrow, self.left_eye,
                         self.right_eye, self.nose_bridge, self.nose_tip, self.top_lip,
                         self.bottom_lip, self.outer_shape]:
            trans_landmark = np.array([[p.x, p.y] for p in landmark], dtype=np.float32)
            trans_landmark = cv2.perspectiveTransform(np.array([trans_landmark]), matrix)
            out.append(trans_landmark)

        return Landmarks(*out)


class Face(NamedTuple):
    """Models a detected face in an image."""
    rect: Rect
    landmarks: Landmarks

    def weighting_previous(self, face: 'Face', alpha: float) -> 'Face':
        return Face(_rect_exp_avg(self.rect, face.rect, alpha),
                    _landmarks_exp_avg(self.landmarks, face.landmarks, alpha))


# Private


def _exp_avg(cur: int, history: int, alpha: float) -> int:
    return int(cur * alpha + history * (1.0 - alpha))


def _point_exp_avg(c: Point, h: Point, a: float) -> Point:
    return Point(_exp_avg(c.x, h.x, a), _exp_avg(c.y, h.y, a))


def _rect_exp_avg(c: Rect, h: Rect, a: float) -> Rect:
    return Rect(
        x=_exp_avg(c.x, h.x, a),
        y=_exp_avg(c.y, h.y, a),
        width=_exp_avg(c.width, h.width, a),
        height=_exp_avg(c.height, h.height, a)
    )


def _point_list_exp_avg(c: List[Point], h: List[Point], a: float) -> List[Point]:
    avg = []

    for idx, pt in enumerate(c):
        avg.append(_point_exp_avg(pt, h[idx], a))

    return avg


def _landmarks_exp_avg(c: Landmarks, h: Landmarks, a: float) -> Landmarks:
    return Landmarks(
        chin=_point_list_exp_avg(c.chin, h.chin, a),
        left_eyebrow=_point_list_exp_avg(c.left_eyebrow, h.left_eyebrow, a),
        right_eyebrow=_point_list_exp_avg(c.right_eyebrow, h.right_eyebrow, a),
        left_eye=_point_list_exp_avg(c.left_eye, h.left_eye, a),
        right_eye=_point_list_exp_avg(c.right_eye, h.right_eye, a),
        nose_bridge=_point_list_exp_avg(c.nose_bridge, h.nose_bridge, a),
        nose_tip=_point_list_exp_avg(c.nose_tip, h.nose_tip, a),
        top_lip=_point_list_exp_avg(c.top_lip, h.top_lip, a),
        bottom_lip=_point_list_exp_avg(c.bottom_lip, h.bottom_lip, a),
        outer_shape=_point_list_exp_avg(c.outer_shape, h.outer_shape, a)
    )
