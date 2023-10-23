import math
import struct

import glwindow
import zengl


class BoxGrid:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', samples=samples)
        self.depth = self.ctx.image(size, 'depth24plus', samples=samples)
        self.output = self.image if self.image.samples == 1 else self.ctx.image(size, 'rgba8unorm')

        model = struct.pack('3f3f3f', -0.866, -0.5, 0.0, 0.866, -0.5, 0.0, 0.0, 1.0, 0.0)

        self.vertex_buffer = self.ctx.buffer(model)
        self.uniform_buffer = self.ctx.buffer(size=64, uniform=True)
        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                };

                vec3 vertices[36] = vec3[](
                    vec3(-0.5, -0.5, -0.5),
                    vec3(-0.5, 0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(-0.5, 0.5, -0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(-0.5, 0.5, -0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(-0.5, 0.5, -0.5)
                );

                out vec3 v_color;

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                float hash13(vec3 p3) {
                    p3 = fract(p3 * 0.1031);
                    p3 += dot(p3, p3.zyx + 31.32);
                    return fract((p3.x + p3.y) * p3.z);
                }

                void main() {
                    int N = 10;
                    float px = float(gl_InstanceID % N);
                    float py = float(gl_InstanceID / N % N);
                    float pz = float(gl_InstanceID / N / N % N);
                    vec3 position = vec3(px, py, pz) - float(N - 1) / 2.0;
                    float scale = 0.1;
                    gl_Position = mvp * vec4(position + vertices[gl_VertexID] * scale, 1.0);
                    v_color = hsv2rgb(vec3(hash13(position), 1.0, 0.5));
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec3 v_color;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(v_color, 1.0);
                    out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
                }
            ''',
            layout=[
                {
                    'name': 'Common',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.uniform_buffer,
                }
            ],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            cull_face='back',
            vertex_count=36,
            instance_count=1000,
        )

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time) * 5.0, math.sin(self.time) * 5.0, math.sin(self.time * 0.7) * 2.0)
        camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=self.aspect, fov=45.0)
        self.uniform_buffer.write(camera)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()
        if self.image != self.output:
            self.image.blit(self.output)


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()
        self.scene = BoxGrid(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
