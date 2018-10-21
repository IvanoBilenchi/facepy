import cv2.cv2 as cv2
import numpy as np
from enum import Enum

from face_auth.config import Renderer
from face_auth.model.geometry import Size

Config = Renderer.Label


class LabelPosition(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


class LabelRenderer:

    def __init__(self, position: LabelPosition, width: int = 0) -> None:
        self.position = position
        self.width = width
        self.padding = 10

    def render(self, image: np.array, text: str, color=Config.FONT_COLOR) -> None:
        size = Size.of_image(image)
        x, y = self.__compute_x(size.width, text), self.__compute_y(size.height)

        cv2.putText(image, text, (x, y), Config.FONT, Config.FONT_SCALE, color,
                    Config.FONT_THICKNESS, Config.LINE_TYPE)

    def size_to_fit(self, text: str) -> None:
        self.width = self.__compute_width(text)

    # Private

    def __compute_width(self, text: str) -> int:
        return cv2.getTextSize(text, Config.FONT, Config.FONT_SCALE,
                               Config.FONT_THICKNESS)[0][0]

    def __compute_x(self, image_width: int, text: str) -> int:
        if self.position == LabelPosition.TOP_LEFT or self.position == LabelPosition.BOTTOM_LEFT:
            return self.padding
        else:
            width = self.width if self.width > 0 else self.__compute_width(text)
            return image_width - width - self.padding

    def __compute_y(self, image_height: int) -> int:
        if self.position == LabelPosition.TOP_LEFT or self.position == LabelPosition.TOP_RIGHT:
            return Config.FONT_HEIGHT + self.padding
        else:
            return image_height - self.padding
