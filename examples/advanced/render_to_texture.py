import sys

import pygame
import zengl
import zengl_extras
from crate import Crate
from monkey import Monkey


class App:
    def __init__(self):
        zengl_extras.init()

        pygame.init()
        pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

        window_size = pygame.display.get_window_size()        
        self.ctx = zengl.context()
        self.scene1 = Monkey(window_size)
        self.scene1.image.clear_value = (0.15, 0.15, 0.15, 1.0)
        self.scene2 = Crate(window_size, texture=self.scene1.output)

    def update(self):
        self.ctx.new_frame()
        self.scene1.render()
        self.scene2.render()
        self.scene2.output.blit()
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

