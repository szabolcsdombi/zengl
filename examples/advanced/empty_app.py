import sys

import pygame
import zengl
import zengl_extras


class Scene:
    def __init__(self, size):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, "rgba8unorm", bytes(size[0] * size[1] * 4))
        self.output = self.image

    def render(self):
        pass


class App:
    def __init__(self):
        zengl_extras.init()
        pygame.init()
        pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)
        self.ctx = zengl.context()
        self.scene = Scene(pygame.display.get_window_size())

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
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
