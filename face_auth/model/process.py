import numpy as np
from os import path
from time import perf_counter_ns
from typing import Callable, List, NamedTuple, Optional

from . import img
from face_auth.config import Paths


class Step(NamedTuple):
    """Models a processing step."""
    name: str
    run: Callable[[np.array], np.array]


class Pipeline:
    """Models a frame processing pipeline."""

    @classmethod
    def execute(cls, name: str, frame: np.array, debug: bool = False,
                steps: Optional[List[Step]] = None) -> np.array:
        return Pipeline(name, debug, steps).run(frame)

    def __init__(self, name: str, debug: bool, steps: Optional[List[Step]]) -> None:
        self.name = name
        self.steps = steps if steps is not None else []
        self.debug = debug

    def run(self, frame: np.array) -> np.array:
        return self.__run_debug(frame) if self.debug else self.__run(frame)

    # Private methods

    def __run_debug(self, frame: np.array) -> np.array:
        self.__save(frame, 0, 'start')

        for n, step in enumerate(self.steps, 1):
            start = perf_counter_ns()
            frame = step.run(frame)
            print('{} - {}. {}: {:.2f} ms'.format(self.name, n, step.name,
                                                  (perf_counter_ns() - start) / 1000000))
            self.__save(frame, n, step.name)
        return frame

    def __run(self, frame: np.array) -> np.array:
        for step in self.steps:
            frame = step.run(frame)
        return frame

    def __save(self, frame: np.array, n: int, step_name: str) -> None:
        file_name = '_'.join(self.name.split() + [str(n)] + step_name.split()) + '.png'
        abs_path = path.join(Paths.USER_DIR, file_name.lower())
        img.save(frame, abs_path)
