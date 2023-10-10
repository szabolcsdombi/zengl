from typing import Tuple

from context import Context


class SimpleFramebuffer:
    def __init__(self, size: Tuple[int, int]):
        ctx = Context.context
        self.image = ctx.image(size, 'rgba8unorm')
        self.depth = ctx.image(size, 'depth24plus')
        self.framebuffer = [self.image, self.depth]

    def clear(self, red: float, green: float, blue: float):
        self.image.clear_value = (red, green, blue, 1.0)
        self.image.clear()
        self.depth.clear()
