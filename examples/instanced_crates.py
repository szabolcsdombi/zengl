import struct

import numpy as np
import zengl
from objloader import Obj
from PIL import Image

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

instance_buffer = ctx.buffer(np.array([
    np.repeat(np.linspace(-20.0, 20.0, 30), 30),
    np.tile(np.linspace(-20.0, 20.0, 30), 30),
    np.random.normal(0.0, 0.2, 30 * 30),
]).T.astype('f4').tobytes())

img = Image.open(assets.get('crate.png')).convert('RGBA')
texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())
texture.mipmaps()

uniform_buffer = ctx.buffer(size=80)

crate = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec2 in_text;
        layout (location = 3) in vec3 in_pos;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec2 v_text;

        void main() {
            v_vert = in_pos + in_vert;
            gl_Position = mvp * vec4(v_vert, 1.0);
            v_norm = in_norm;
            v_text = in_text;
        }
    ''',
    fragment_shader='''
        #version 330

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
            'min_filter': 'linear_mipmap_linear',
            'mag_filter': 'linear',
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        *zengl.bind(instance_buffer, '3f /i', 3),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
    instance_count=30 * 30,
)

camera = zengl.camera((3.0, 2.0, 1.5), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)

uniform_buffer.write(camera)
uniform_buffer.write(struct.pack('3f4x', 3.0, 2.0, 1.5), offset=64)

while window.update():
    z = np.frombuffer(instance_buffer.map(), 'f4').reshape(-1, 3)
    z[:, 2] += np.random.normal(0.0, 0.01, z.shape[0])
    instance_buffer.unmap()

    image.clear()
    depth.clear()
    crate.render()
    image.blit()
