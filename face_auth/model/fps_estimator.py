from time import perf_counter_ns


class FPSEstimator:
    """Estimates FPS."""

    # Public methods

    def __init__(self) -> None:
        self.fps = 0.0
        self.__frame_count = 0
        self.__frame_timestamp = perf_counter_ns()

    def tick_and_compute_fps(self) -> float:
        self.__frame_count += 1

        current_ns = perf_counter_ns()
        delta = current_ns - self.__frame_timestamp

        if delta >= 500000000:
            self.fps = self.__frame_count / delta * 1000000000
            self.__frame_count = 0
            self.__frame_timestamp = current_ns

        return self.fps
