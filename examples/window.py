import pyglet


class Window(pyglet.window.Window):
    def __init__(self, width, height):
        self.size = width, height
        self.aspect = width / height
        self.time = 0.0
        config = pyglet.gl.Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            double_buffer=True,
            depth_size=0,
            samples=0,
        )
        super().__init__(width=width, height=height, config=config)

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        pass

    def render(self, func):
        def wrapper(dt):
            self.time += dt
            func()
        pyglet.clock.schedule_interval(wrapper, 1.0 / 60.0)

    @staticmethod
    def run():
        pyglet.app.run()
