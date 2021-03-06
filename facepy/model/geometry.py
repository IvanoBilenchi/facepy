import math
import sys
from dlib import rectangle
from typing import List, NamedTuple, Optional

import cv2.cv2 as cv2
import numpy as np


class Point(NamedTuple):
    """Models a point."""
    x: int = 0
    y: int = 0

    @classmethod
    def to_numpy(cls, pts: List['Point']) -> np.array:
        """Converts a list of n points into a (n, 2) NumPy array."""
        return np.int32([np.asarray(pts)])

    @classmethod
    def from_numpy(cls, pts: np.array) -> List['Point']:
        """Converts a (n, 2) NumPy array into a list of n points."""
        return [Point(p[0], p[1]) for p in pts]

    @classmethod
    def mean(cls, pts: List['Point']) -> 'Point':
        """Returns the mean of a list of points."""
        x, y, n = 0, 0, len(pts)

        if n == 0:
            return Point(0, 0)

        for point in pts:
            x += point.x
            y += point.y

        return Point(x // n, y // n)

    def distance(self, other: 'Point') -> int:
        """Computes the distance from another point."""
        return int(math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


class Size(NamedTuple):
    """Models sizes."""
    width: int = 0
    height: int = 0

    @classmethod
    def of_image(cls, image: np.array) -> 'Size':
        """Returns the size of the specified image."""
        shape = image.shape[:2]
        return Size(shape[1], shape[0])

    @property
    def center(self) -> Point:
        """Returns the center of the rectangle of 'self' size."""
        return Point(self.width // 2, self.height // 2)


class Rect(NamedTuple):
    """Models a rectangle."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @classmethod
    def with_size(cls, size: Size) -> 'Rect':
        """Returns a rectangle of the specified size with (x, y) = (0, 0)."""
        return Rect(0, 0, *size)

    @classmethod
    def from_dlib_rect(cls, rect: rectangle) -> 'Rect':
        """Converts a Dlib rectangle into a Rect."""
        return Rect(rect.left(), rect.top(), rect.width(), rect.height())

    @classmethod
    def nearest_to_center(cls, rects: List['Rect'], point: Point) -> Optional['Rect']:
        """Returns the rect in the list whose center is nearest to 'point'."""
        min_distance = sys.maxsize
        nearest = None

        for rect in rects:
            distance = rect.center.distance(point)

            if distance < min_distance:
                min_distance = distance
                nearest = rect

        return nearest

    @property
    def top_left(self) -> Point:
        """Returns the top-left point of this rectangle."""
        return Point(self.x, self.y)

    @property
    def bottom_right(self) -> Point:
        """Returns the bottom-right point of this rectangle."""
        return Point(self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Point:
        """Returns the center of this rectangle."""
        return Point(self.x + self.width // 2, self.y + self.height // 2)

    @property
    def size(self) -> Size:
        """Returns the size of this rectangle."""
        return Size(self.width, self.height)

    def scaled(self, scale_factor: int) -> 'Rect':
        """Scales this rectangle by the specified scale factor."""
        return Rect(self.x * scale_factor, self.y * scale_factor,
                    self.width * scale_factor, self.height * scale_factor)

    def shrunk_to_square(self) -> 'Rect':
        """Shrinks this rectangle to a square having side = min(width, height)."""
        x, y, w, h = self.x, self.y, self.width, self.height

        if w == h:
            return self

        return Rect(x + (w - h) // 2, y, h, h) if w > h else Rect(x, y + (h - w) // 2, w, w)

    def expanded_to_square(self) -> 'Rect':
        """Expands this rectangle to a square having side = max(width, height)."""
        x, y, w, h = self.x, self.y, self.width, self.height

        if w == h:
            return self

        return Rect(x, y - (w - h) // 2, w, w) if w > h else Rect(x - (h - w) // 2, y, h, h)

    def to_dlib_rect(self) -> rectangle:
        """Converts this rectangle into a dlib rectangle."""
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
        """Converts the Dlib pose estimator output into a 'Landmarks' object."""
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
        )

    @property
    def all(self) -> List[Point]:
        """Returns all the landmarks as a list of points."""
        return (self.chin + self.left_eyebrow + self.right_eyebrow + self.left_eye +
                self.right_eye + self.nose_bridge + self.nose_tip + self.top_lip + self.bottom_lip)

    @property
    def thin_shape(self) -> List[Point]:
        """Returns a 'thin' shape of the face, useful for masking purposes."""
        return self.left_eyebrow[:3] + self.right_eyebrow[-3:] + self.chin[11:4:-1]

    @property
    def outer_shape(self) -> List[Point]:
        """Returns the outer shape of the face."""
        return self.left_eyebrow[1:3] + self.right_eyebrow[-3:-1] + self.chin[::-1]

    def rect(self) -> Rect:
        """Returns the rect enclosing the facial landmarks."""
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
        """Returns the square enclosing the facial landmarks."""
        return Rect.expanded_to_square(self.rect())

    def alignment_matrix(self) -> np.array:
        """
        Returns a transformation matrix which can be used to align images
        based on the position of the eyes.
        """
        # Get the center of the eyes
        left_eye = Point.mean(self.left_eye)
        right_eye = Point.mean(self.right_eye)
        face_square = self.square()

        # Compute tilt
        delta_y = right_eye.y - left_eye.y
        delta_x = right_eye.x - left_eye.x
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Normalized eye positions
        out_left_eye_x, out_left_eye_y = 0.3, 0.2
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

    def pose_is_frontal(self) -> bool:
        """Checks if these landmarks correspond to a frontal pose."""
        return self.__check_pose(1.2, -0.5, 0.2)

    def pose_is_valid(self) -> bool:
        """
        Checks if these landmarks correspond to a 'valid' pose.
        Allows discarding very tilted/imprecise landmark detections.
        """
        return self.__check_pose(1.0, -2.0, 0.6)

    def __check_pose(self, alpha: float, beta: float, gamma: float) -> bool:
        """
        Parametric pose checking function. Alpha, beta and gamma account for upwards, downwards
        and sideways tilt of the head.
        """
        nose_tip_l, nose_tip_r = self.nose_tip[0], self.nose_tip[-1]
        nose_bridge_t, nose_bridge_b = self.nose_bridge[0], self.nose_bridge[-1]

        nose_w = nose_tip_l.distance(nose_tip_r)
        nose_h = nose_bridge_t.distance(nose_bridge_b)

        # Discard faces tilted upwards.
        # Check ratio between nose height and width.
        if nose_h / nose_w < alpha:
            return False

        # Discard faces tilted downwards.
        # Use cross product to detect if the lowest point of the nose bridge
        # crosses the line at the base of the nose tip.
        v1 = Point(nose_tip_r.x - nose_tip_l.x, nose_tip_r.y - nose_tip_l.y)
        v2 = Point(nose_tip_r.x - nose_bridge_b.x, nose_tip_r.y - nose_bridge_b.y)
        v1_len = math.sqrt(v1.x ** 2 + v1.y ** 2)

        if (v1.x * v2.y - v1.y * v2.x) / v1_len < beta:
            return False

        l_eye, r_eye = self.left_eye[3], self.right_eye[0]
        nose_eye_ratio = nose_bridge_t.distance(l_eye) / nose_bridge_t.distance(r_eye)

        # Discard faces tilted sideways.
        # Check ratio between eyes and base of the nose bridge.
        return 1.0 - gamma < nose_eye_ratio < 1.0 + gamma


class Face(NamedTuple):
    """Models a detected face in an image."""
    rect: Rect
    landmarks: Landmarks

    def weighting_previous(self, face: 'Face', alpha: float) -> 'Face':
        """
        Returns the exponential average between this object and another Face object
        representing past history.
        """
        return Face(_rect_exp_avg(self.rect, face.rect, alpha),
                    _landmarks_exp_avg(self.landmarks, face.landmarks, alpha))


# Private - Exponential average functions


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
    )
