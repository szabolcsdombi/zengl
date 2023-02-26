import pyglet

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False


class Window(pyglet.window.Window):
    def __init__(self, width, height):
        self.time = 0.0
        self.alive = True
        self.mouse = (0, 0)
        config = pyglet.gl.Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            double_buffer=False,
            depth_size=0,
            samples=0,
        )
        super().__init__(width=width, height=height, config=config, vsync=True)
        width, height = self.get_framebuffer_size()
        self.size = (width, height)
        self.aspect = width / height

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        pass

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (x, y)

    def on_close(self):
        self.alive = False

    def update(self):
        self.flip()
        self.dispatch_events()
        self.time += 1.0 / 60.0
        return self.alive

    @staticmethod
    def run():
        pyglet.app.render()
