class Rect:
    """Models a rectangle."""

    @property
    def top_left(self) -> (int, int):
        return self.x, self.y

    @property
    def bottom_right(self) -> (int, int):
        return self.x + self.width, self.y + self.height

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def scale(self, scale_factor: int):
        self.x *= scale_factor
        self.y *= scale_factor
        self.width *= scale_factor
        self.height *= scale_factor
