import glwindow
import numpy as np
import zengl


class Blending:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()

        self.image = self.ctx.image(size, "rgba8unorm", samples=samples)
        self.depth = self.ctx.image(size, "depth24plus", samples=samples)
        self.output = self.image if self.image.samples == 1 else self.ctx.image(size, "rgba8unorm")

        self.uniform_buffer = self.ctx.buffer(size=16, uniform=True)

        triangle = np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.5],
                [-0.5, 0.86, 0.0, 1.0, 0.0, 0.5],
                [-0.5, -0.86, 0.0, 0.0, 1.0, 0.5],
            ],
            "f4",
        )

        self.vertex_buffer = self.ctx.buffer(triangle)

        self.pipeline = self.ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    vec2 scale;
                    float rotation;
                };

                layout (location = 0) in vec2 in_vert;
                layout (location = 1) in vec4 in_color;

                out vec4 v_color;

                void main() {
                    float r = rotation * (0.5 + float(gl_InstanceID) * 0.05);
                    mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
                    gl_Position = vec4((rot * in_vert) * scale, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;

                in vec4 v_color;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(v_color);
                    out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
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
            blend={
                "enable": True,
                "src_color": "src_alpha",
                "dst_color": "one_minus_src_alpha",
            },
            framebuffer=[self.image],
            topology="triangles",
            vertex_buffers=zengl.bind(self.vertex_buffer, "2f 4f", 0, 1),
            vertex_count=3,
            instance_count=10,
        )

        self.aspect = size[0] / size[1]
        self.scale = 0.5
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        self.image.clear()
        self.uniform_buffer.write(np.array([self.scale, self.scale * self.aspect, self.time, 0.0], "f4"))
        self.pipeline.render()
        if self.image != self.output:
            self.image.blit(self.output)


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()
        self.logo = Blending(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.logo.render()
        self.logo.output.blit()
        self.ctx.end_frame()


if __name__ == "__main__":
    glwindow.run(App)
