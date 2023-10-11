import glwindow
import zengl
from blending import Blending
from crate import Crate


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene1 = Blending((256, 256), samples=1)
        self.scene1.image.clear_value = (0.15, 0.15, 0.15, 1.0)
        self.scene1.scale = 0.8
        self.scene2 = Crate(self.wnd.size, texture=self.scene1.image)

    def update(self):
        self.ctx.new_frame()
        self.scene1.render()
        self.scene2.render()
        self.scene2.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
