# TODO:
# Blending, Grass, WireframeTerrain, ZenglLogo examples are no longer classes

import glwindow
import zengl
from blending import Blending
from grass import Grass
from wireframe_terrain import WireframeTerrain
from zengl_logo import Logo


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.half_size = self.wnd.size[0] // 2, self.wnd.size[1] // 2
        self.ctx = zengl.context()
        self.scene1 = Logo(self.half_size)
        self.scene2 = WireframeTerrain(self.half_size)
        self.scene3 = Grass(self.half_size, 100)
        self.scene4 = Blending(self.half_size)

    def update(self):
        self.ctx.new_frame()
        w, h = self.half_size
        self.scene1.render()
        self.scene2.render()
        self.scene3.render()
        self.scene4.render()
        self.scene1.output.blit(None, (0, h, w, h))
        self.scene2.output.blit(None, (w, h, w, h))
        self.scene3.output.blit(None, (0, 0, w, h))
        self.scene4.output.blit(None, (w, 0, w, h))
        self.ctx.end_frame()


if __name__ == "__main__":
    glwindow.run(App)
