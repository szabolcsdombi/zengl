import os

import glwindow
import zengl


class Scene:
    def __init__(self, size):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', os.urandom(size[0] * size[1] * 4))


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene = Scene(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
