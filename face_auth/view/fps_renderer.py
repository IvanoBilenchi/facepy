import numpy as np
from .label_renderer import LabelPosition, LabelRenderer


class FPSRenderer:

    def __init__(self) -> None:
        self.__label_renderer = LabelRenderer(LabelPosition.TOP_RIGHT)
        self.__label_renderer.size_to_fit('FPS: 99.99')

    def render(self, frame: np.array, fps: float) -> None:
        self.__label_renderer.render(frame, 'FPS: {:.2f}'.format(fps))
