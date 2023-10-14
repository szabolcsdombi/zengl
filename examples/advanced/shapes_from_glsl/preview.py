import math
import struct

import glwindow
import zengl


class Preview:
    def __init__(self, size, shapes, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', samples=samples)
        self.depth = self.ctx.image(size, 'depth24plus', samples=samples)
        self.output = self.image if self.image.samples == 1 else self.ctx.image(size, 'rgba8unorm')

        self.uniform_buffer = self.ctx.buffer(size=96, uniform=True)
        self.pipelines = [
            shape([self.image, self.depth], self.uniform_buffer)
            for shape in shapes
        ]

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time * 0.4) * 5.0, math.sin(self.time * 0.4) * 5.0, 2.0)
        camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=self.aspect, fov=45.0)
        light = eye[0], eye[1], eye[2] + 2.0
        self.uniform_buffer.write(struct.pack('64s3f4x3f4x', camera, *eye, *light))
        self.image.clear_value = (0.2, 0.2, 0.2, 1.0)
        self.image.clear()
        self.depth.clear()
        for pipeline in self.pipelines:
            pipeline.render()
        if self.image != self.output:
            self.image.blit(self.output)


def show(shapes):
    class App:
        def __init__(self):
            self.wnd = glwindow.get_window()
            self.ctx = zengl.context(glwindow.get_loader())
            self.scene = Preview(self.wnd.size, shapes)

        def update(self):
            self.ctx.new_frame()
            self.scene.render()
            self.scene.output.blit()
            self.ctx.end_frame()

    glwindow.run(App)


if __name__ == '__main__':
    from grid import Grid

    show([Grid])
