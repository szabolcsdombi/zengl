import math
import struct

import pyglet
import zengl
from meshtools import obj
from zengl_extras import assets

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False


def load_texture(name):
    ctx = zengl.context()
    img = pyglet.image.load(assets.get(name))
    return ctx.image((img.width, img.height), 'rgba8unorm', img.get_data('RGBA', img.pitch))


def load_model(name):
    ctx = zengl.context()
    with open(assets.get(name)) as f:
        model = obj.parse_obj(f.read(), 'vnt')
    return ctx.buffer(model)


window_size = (1280, 720)

config = pyglet.gl.Config(
    major_version=3,
    minor_version=3,
    forward_compatible=True,
    double_buffer=True,
    depth_size=0,
    samples=0,
)

window = pyglet.window.Window(*window_size, resizable=False, config=config, vsync=True)

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)

texture = load_texture('crate.png')
vertex_buffer = load_model('box.obj')

uniform_buffer = ctx.buffer(size=80, uniform=True)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
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
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;
        in vec2 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = clamp(dot(normalize(light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.6 + 0.4;
            out_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
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


class g:
    time = 0.0


def on_draw(dt=0.0):
    g.time += dt
    ctx.new_frame()
    eye = (math.cos(g.time * 0.6) * 3.0, math.sin(g.time * 0.6) * 3.0, 1.5)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=16.0 / 9.0, fov=45.0)
    uniform_buffer.write(struct.pack('64s3f4x', camera, *eye))
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()


pyglet.clock.schedule_interval(on_draw, 1.0 / 60.0)
pyglet.app.run()
