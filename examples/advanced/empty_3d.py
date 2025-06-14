import math
import struct
import sys

import pygame
import zengl
import zengl_extras


class Scene:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, "rgba8unorm", samples=samples)
        self.depth = self.ctx.image(size, "depth24plus", samples=samples)
        self.output = self.image if self.image.samples == 1 else self.ctx.image(size, "rgba8unorm")

        model = struct.pack("3f3f3f", -0.866, -0.5, 0.0, 0.866, -0.5, 0.0, 0.0, 1.0, 0.0)

        self.vertex_buffer = self.ctx.buffer(model)
        self.uniform_buffer = self.ctx.buffer(size=64, uniform=True)
        self.pipeline = self.ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                };

                layout (location = 0) in vec3 in_vert;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(1.0, 1.0, 1.0, 1.0);
                }
            """,
            layout=[
                {
                    "name": "Common",
                    "binding": 0,
                },
            ],
            resources=[
                {
                    "type": "uniform_buffer",
                    "binding": 0,
                    "buffer": self.uniform_buffer,
                },
            ],
            framebuffer=[self.image, self.depth],
            topology="triangles",
            cull_face="back",
            vertex_buffers=zengl.bind(self.vertex_buffer, "3f", 0),
            vertex_count=self.vertex_buffer.size // zengl.calcsize("3f"),
        )

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time) * 5.0, math.sin(self.time) * 5.0, 2.0)
        camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=self.aspect, fov=45.0)
        self.uniform_buffer.write(camera)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()
        if self.image != self.output:
            self.image.blit(self.output)


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
