import glwindow
import zengl


class Scene:
    def __init__(self, size):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', bytes(size[0] * size[1] * 4))
        self.output = self.image

    def render(self):
        pass


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()
        self.scene = Scene(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
