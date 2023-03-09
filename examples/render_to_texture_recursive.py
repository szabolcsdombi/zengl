import math
import struct

import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

texture_0 = ctx.image((4, 4), 'rgba8unorm', b'\x0f' * 64)
depth_1 = ctx.image((128, 128), 'depth24plus')
texture_1 = ctx.image((128, 128), 'rgba8unorm')
texture_1.clear_value = (0.4, 0.4, 0.4, 1.0)
depth_2 = ctx.image((256, 256), 'depth24plus')
texture_2 = ctx.image((256, 256), 'rgba8unorm')
texture_2.clear_value = (0.6, 0.6, 0.6, 1.0)
depth_3 = ctx.image((512, 512), 'depth24plus')
texture_3 = ctx.image((512, 512), 'rgba8unorm')
texture_3.clear_value = (0.8, 0.8, 0.8, 1.0)

uniform_buffer = ctx.buffer(size=80)


def crate_pipeline(source_image, target_image, target_depth):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            uniform Common {
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
            #version 330 core

            uniform Common {
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
                'image': source_image,
            }
        ],
        framebuffer=[target_image, target_depth],
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
    )


crate_1 = crate_pipeline(texture_0, texture_1, depth_1)
crate_2 = crate_pipeline(texture_1, texture_2, depth_2)
crate_3 = crate_pipeline(texture_2, texture_3, depth_3)
crate_4 = crate_pipeline(texture_3, image, depth)

while window.update():
    ctx.new_frame()
    x, y = math.sin(window.time * 0.5) * 2.0, math.cos(window.time * 0.5) * 2.0
    camera = zengl.camera((x, y, 0.8), (0.0, 0.0, -0.15), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)
    uniform_buffer.write(struct.pack('3f4x', x, y, 1.5), offset=64)

    texture_1.clear()
    depth_1.clear()
    crate_1.render()

    texture_2.clear()
    depth_2.clear()
    crate_2.render()

    texture_3.clear()
    depth_3.clear()
    crate_3.render()

    image.clear()
    depth.clear()
    crate_4.render()

    image.blit()
    ctx.end_frame()
