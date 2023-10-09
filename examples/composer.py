import glwindow
import zengl

from zengl_logo import Logo
from grass import Grass
from blending import Blending
from box_grid import BoxGrid


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.half_size = self.wnd.size[0] // 2, self.wnd.size[1] // 2
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene1 = Logo(self.half_size)
        self.scene2 = BoxGrid(self.half_size)
        self.scene3 = Grass(100, self.half_size)
        self.scene4 = Blending(self.half_size)

    def update(self):
        self.ctx.new_frame()
        w, h = self.half_size
        self.scene1.render()
        self.scene2.render()
        self.scene3.render()
        self.scene4.render()
        self.scene1.image.blit(None, (0, h, w, h))
        self.scene2.image.blit(None, (w, h, w, h))
        self.scene3.image.blit(None, (0, 0, w, h))
        self.scene4.image.blit(None, (w, 0, w, h))
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
