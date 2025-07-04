import sys

import pygame
import zengl
import zengl_extras
from box_grid import BoxGrid
from crate import Crate
from monkey import Monkey
from sprites import Sprites


class App:
    def __init__(self):
        zengl_extras.init()
        pygame.init()
        pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)
        window_size = pygame.display.get_window_size()

        self.half_size = window_size[0] // 2, window_size[1] // 2
        self.ctx = zengl.context()
        self.scene1 = Monkey(self.half_size)
        self.scene2 = Crate(self.half_size)
        self.scene3 = Sprites(self.half_size, 100)
        self.scene4 = BoxGrid(self.half_size)

    def update(self):
        self.ctx.new_frame()
        w, h = self.half_size
        self.scene1.render()
        self.scene2.render()
        self.scene3.render()
        self.scene4.render()
        self.scene1.output.blit(None, (0, h), (w, h))
        self.scene2.output.blit(None, (w, h), (w, h))
        self.scene3.output.blit(None, (0, 0), (w, h))
        self.scene4.output.blit(None, (w, 0), (w, h))
        self.ctx.end_frame()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.update()
            pygame.display.flip()


if __name__ == "__main__":
    App().run()
