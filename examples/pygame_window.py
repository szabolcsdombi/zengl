import math
import struct

import pygame as pg
import zengl
from objloader import Obj
from PIL import Image

import assets

window_size = (1280, 720)
window_aspect = 16.0 / 9.0

pg.init()

pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

pg.display.set_mode(window_size, pg.OPENGL | pg.DOUBLEBUF)

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

img = Image.open(assets.get('crate.png')).convert('RGBA')
texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())

uniform_buffer = ctx.buffer(size=80)

crate = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec2 in_text;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec2 v_text;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
            v_text = in_text;
        }
    ''',
    fragment_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        layout (binding = 0) uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;
        in vec2 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = clamp(dot(normalize(light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.6 + 0.4;
            out_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

frames = 0
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    x, y = math.sin(frames * 0.02) * 3.0, math.cos(frames * 0.02) * 3.0
    camera = zengl.camera((x, y, 1.5), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)

    uniform_buffer.write(camera)
    uniform_buffer.write(struct.pack('3f4x', x, y, 1.5), offset=64)

    image.clear()
    depth.clear()
    crate.render()
    image.blit()

    pg.display.flip()
    pg.time.wait(10)
    frames += 1

pg.quit()
