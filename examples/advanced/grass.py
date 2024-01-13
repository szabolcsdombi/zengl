import math

import glwindow
import zengl


def grass_mesh():
    verts = []
    for i in range(7):
        u = i / 7
        v = math.sin(u * u * (math.pi - 1.0) + 1.0)
        verts.append((-v * 0.03, u * u * 0.2, u))
        verts.append((v * 0.03, u * u * 0.2, u))
    verts.append((0.0, 0.2, 1.0))
    verts = ",".join("vec3(%.8f, %.8f, %.8f)" % x for x in verts)
    return f"vec3 grass[15] = vec3[]({verts});"


class Grass:
    def __init__(self, size, count, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, "rgba8unorm", samples=samples)
        self.depth = self.ctx.image(size, "depth24plus", samples=samples)
        self.output = self.image if self.image.samples == 1 else self.ctx.image(size, "rgba8unorm")

        self.ubo_data = bytearray(64)
        self.uniform_buffer = self.ctx.buffer(self.ubo_data, uniform=True)
        self.pipeline = self.ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;

                #include "N"
                #include "grass"

                vec4 hash41(float p) {
                    vec4 p4 = fract(vec4(p) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
                    p4 += dot(p4, p4.wzxy + 33.33);
                    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
                }

                float hash11(float p) {
                    p = fract(p * 0.1031);
                    p *= p + 33.33;
                    p *= p + p;
                    return fract(p);
                }

                layout (std140) uniform Common {
                    mat4 mvp;
                };

                out vec2 v_data;

                void main() {
                    vec3 v = grass[gl_VertexID];
                    vec4 data = hash41(float(gl_InstanceID));
                    vec2 cell = vec2(float(gl_InstanceID % N), float(gl_InstanceID / N));
                    float height = (sin(cell.x * 0.1) + cos(cell.y * 0.1)) * 0.2;
                    float scale = 0.9 + hash11(float(gl_InstanceID)) * 0.2;
                    data.xy = (data.xy + cell - float(N / 2)) * 0.1;
                    data.z *= 6.283184;
                    vec3 vert = vec3(
                        data.x + cos(data.z) * v.x + sin(data.z) * v.y,
                        data.y + cos(data.z) * v.y - sin(data.z) * v.x,
                        height + v.z
                    );
                    vert *= scale;
                    gl_Position = mvp * vec4(vert, 1.0);
                    v_data = vec2(data.w, v.z);
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;

                in vec2 v_data;

                layout (location = 0) out vec4 out_color;

                void main() {
                    vec3 yl = vec3(0.63, 1.0, 0.3);
                    vec3 gn = vec3(0.15, 0.83, 0.3);
                    out_color = vec4((yl + (gn - yl) * v_data.x) * v_data.y, 1.0);
                }
            """,
            includes={
                "N": f"const int N = {count};",
                "grass": grass_mesh(),
            },
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
            topology="triangle_strip",
            instance_count=count * count,
            vertex_count=15,
        )

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time * 0.2) * 12.0, math.sin(self.time * 0.2) * 12.0, 4.0)
        self.ubo_data[:] = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=self.aspect, fov=45.0)
        self.uniform_buffer.write(self.ubo_data)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()
        if self.image != self.output:
            self.image.blit(self.output)


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()
        self.scene = Grass(self.wnd.size, 200)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == "__main__":
    glwindow.run(App)
