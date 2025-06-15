import gzip
import struct
import sys
from math import cos, sin

import assets
import pygame
import zengl
import zengl_extras
from objloader import Obj

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()


class Cubemap:
    def __init__(self, size):
        self.depth = ctx.image((size, size), 'depth24plus')
        self.image = ctx.image((size, size), 'rgba8unorm', cubemap=True)
        self.pipelines = []

    def pipeline(self, vertex_buffer):
        row = [
            ctx.pipeline(
                vertex_shader='''
                    #version 300 es
                    precision highp float;

                    const float znear = 0.1;
                    const float zfar = 100.0;
                    const float f1 = (zfar + znear) / (zfar - znear);
                    const float f2 = -2.0 * zfar * znear / (zfar - znear);

                    mat4 mvp[6] = mat4[](
                        mat4(0.0, 0.0, f1, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                        mat4(0.0, 0.0, -f1, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                        mat4(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, f1, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                        mat4(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -f1, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                        mat4(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, f1, 1.0, 0.0, 0.0, f2, 0.0),
                        mat4(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -f1, -1.0, 0.0, 0.0, f2, 0.0)
                    );

                    uniform int face;

                    layout (location = 0) in vec3 in_vertex;
                    layout (location = 1) in vec3 in_normal;

                    out vec3 v_vertex;
                    out vec3 v_normal;

                    void main() {
                        v_vertex = in_vertex;
                        v_normal = in_normal;
                        gl_Position = mvp[face] * vec4(v_vertex, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 300 es
                    precision highp float;

                    in vec3 v_vertex;
                    in vec3 v_normal;

                    layout (location = 0) out vec4 out_color;

                    void main() {
                        vec3 light = vec3(0.0, 0.0, 10.0);
                        float lum = dot(normalize(light - v_vertex), normalize(v_normal)) * 0.7 + 0.3;
                        out_color = vec4(lum, lum, lum, 1.0);
                    }
                ''',
                uniforms={
                    'face': face,
                },
                framebuffer=[self.image.face(layer=face), self.depth],
                topology='triangles',
                cull_face='back',
                vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
                vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
            )
            for face in range(6)
        ]
        self.pipelines.append(row)

    def render(self):
        for i in range(6):
            self.depth.clear()
            self.image.face(i).clear()
            for row in self.pipelines:
                row[i].render()


class Scene:
    def __init__(self):
        self.image = ctx.image(window_size, 'rgba8unorm', samples=4)
        self.depth = ctx.image(window_size, 'depth24plus', samples=4)
        self.image.clear_value = (0.2, 0.2, 0.2, 1.0)
        self.uniform_buffer = ctx.buffer(size=80, uniform=True)
        self.uniform_buffer_data = bytearray(80)
        self.pipelines = []

    def pipeline(self, vertex_buffer, reflecting):
        self.pipelines.append(ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec4 eye;
                };

                layout (location = 0) in vec3 in_vertex;
                layout (location = 1) in vec3 in_normal;

                out vec3 v_vertex;
                out vec3 v_normal;

                void main() {
                    v_vertex = in_vertex;
                    v_normal = in_normal;
                    gl_Position = mvp * vec4(v_vertex, 1.0);
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec4 eye;
                };

                uniform int reflecting;

                uniform samplerCube Texture;

                in vec3 v_vertex;
                in vec3 v_normal;

                layout (location = 0) out vec4 out_color;

                void main() {
                    if (reflecting == 1) {
                        vec3 ray = reflect(v_vertex - eye.xyz, v_normal);
                        out_color = vec4(texture(Texture, ray).rgb * 0.5 + 0.3, 1.0);
                    } else {
                        vec3 light = vec3(0.0, 0.0, 10.0);
                        float lum = dot(normalize(light - v_vertex), normalize(v_normal)) * 0.7 + 0.3;
                        out_color = vec4(lum, lum, lum, 1.0);
                    }
                }
            ''',
            layout=[
                {
                    'name': 'Common',
                    'binding': 0,
                },
                {
                    'name': 'Texture',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.uniform_buffer,
                },
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': cubemap.image,
                },
            ],
            uniforms={
                'reflecting': 1 if reflecting else 0,
            },
            framebuffer=[self.image, self.depth],
            topology='triangles',
            cull_face='back',
            vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
            vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
        ))

    def update(self, t):
        eye = (cos(t) * 5.0, sin(t) * 5.0, 2.0)
        self.uniform_buffer_data[:64] = zengl.camera(eye, (0.0, 0.0, 0.5), aspect=window_aspect, fov=45.0)
        self.uniform_buffer_data[64:76] = struct.pack('3f', *eye)
        self.uniform_buffer.write(self.uniform_buffer_data)

    def render(self):
        self.image.clear()
        self.depth.clear()
        for pipeline in self.pipelines:
            pipeline.render()


model = gzip.decompress(open(assets.get('boxgrid.obj.gz'), 'rb').read())
model = Obj.frombytes(model).pack('vx*4 vy*4 vz*4-2 nx ny nz')
boxgrid_vertex_buffer = ctx.buffer(model)

model = Obj.open(assets.get('blob.obj')).pack('vx vy vz nx ny nz')
monkey_vertex_buffer = ctx.buffer(model)

cubemap = Cubemap(512)
cubemap.pipeline(boxgrid_vertex_buffer)

scene = Scene()
scene.pipeline(boxgrid_vertex_buffer, False)
scene.pipeline(monkey_vertex_buffer, True)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    cubemap.render()
    scene.update(now)
    scene.render()
    scene.image.blit()
    ctx.end_frame()

    pygame.display.flip()
