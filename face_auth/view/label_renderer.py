import cv2.cv2 as cv2
import numpy as np
from enum import Enum
from time import perf_counter_ns

from face_auth.config import Renderer
from face_auth.model.geometry import Size

Config = Renderer.Label


class LabelPosition(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


class TempMessage:

    def __init__(self, text: str, seconds: int) -> None:
        self.text = text
        self._remaining = int(seconds * 1000000000)
        self._last_update = perf_counter_ns()

    def update(self) -> bool:
        timestamp = perf_counter_ns()
        self._remaining = self._remaining - (timestamp - self._last_update)
        self._last_update = timestamp
        return self._remaining > 0


class LabelRenderer:

    def __init__(self, position: LabelPosition, width: int = 0) -> None:
        self.position = position
        self.width = width
        self.text = ''
        self.color = Config.FONT_COLOR
        self._temp_message: TempMessage = None

    def temp_text(self, text: str, seconds: int) -> None:
        self._temp_message = TempMessage(text, seconds)

    def render(self, image: np.array) -> None:
        text = self.__get_text()

        if not text:
            return

        size = Size.of_image(image)
        x, y = self.__compute_x(size.width, text), self.__compute_y(size.height)

        cv2.putText(image, text, (x, y), Config.FONT, Config.FONT_SCALE, self.color,
                    Config.FONT_THICKNESS, Config.LINE_TYPE)

    def size_to_fit(self) -> None:
        self.width = self.__compute_width(self.__get_text())

    # Private

    def __get_text(self) -> str:
        temp_text = None

        if self._temp_message:
            if self._temp_message.update():
                temp_text = self._temp_message.text
            else:
                self._temp_message = None

        return temp_text if temp_text is not None else self.text

    def __compute_width(self, text: str) -> int:
        return cv2.getTextSize(text, Config.FONT, Config.FONT_SCALE,
                               Config.FONT_THICKNESS)[0][0]

    def __compute_x(self, image_width: int, text: str) -> int:
        if self.position == LabelPosition.TOP_LEFT or self.position == LabelPosition.BOTTOM_LEFT:
            return Config.PADDING
        else:
            width = self.width if self.width > 0 else self.__compute_width(text)
            return image_width - width - Config.PADDING

    def __compute_y(self, image_height: int) -> int:
        if self.position == LabelPosition.TOP_LEFT or self.position == LabelPosition.TOP_RIGHT:
            return Config.FONT_HEIGHT + Config.PADDING
        else:
            return image_height - Config.PADDING
