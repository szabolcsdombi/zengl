import pygame as pg


class Window:
    def __init__(self, width, height):
        self.size = width, height
        self.aspect = width / height
        self.time = 0.0
        self.alive = True

        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.size, pg.OPENGL | pg.DOUBLEBUF)

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        pass

    def on_close(self):
        self.alive = False

    def update(self):
        pg.display.flip()
        pg.time.wait(10)
        self.time += 1.0 / 60.0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.alive = False
        return self.alive
