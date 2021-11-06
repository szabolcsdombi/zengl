import math

import ffmpeg
import numpy as np
import zengl
from objloader import Obj

ctx = zengl.context(zengl.loader(headless=True))


width, height = 1280, 720
output = ctx.image((width, height), 'rgba8unorm')
image = ctx.image((width, height), 'rgba8unorm', samples=4)
depth = ctx.image((width, height), 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open('examples/data/box.obj').pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

texture = ctx.image((width, height), 'rgba8unorm')

uniform_buffer = ctx.buffer(size=80)

cube = ctx.pipeline(
    vertex_shader='''
        #version 330

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
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

in_filename = 'examples/data/Jellyfish_720_10s_10MB.mp4'
out_filename = 'examples/data/Jellycube.mp4'

process1 = (
    ffmpeg
    .input(in_filename)
    .vflip()
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    .vflip()
    .output(out_filename, pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

frame = 0

while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break

    texture.write(zengl.rgba(in_bytes, 'rgb'))

    frame += 1
    x, y = math.sin(frame * 0.02) * 3.0, math.cos(frame * 0.02) * 3.0
    camera = zengl.camera((x, y, 2.0), (0.0, 0.0, 0.5), aspect=16.0 / 9.0, fov=45.0)

    uniform_buffer.write(camera)
    uniform_buffer.write(zengl.pack(x, y, 2.0, 0.0), offset=64)

    image.clear()
    depth.clear()
    cube.render()
    image.blit(output)

    out_frame = np.frombuffer(output.read(), 'u1').reshape(width, height, 4)[:, :, :3]

    process2.stdin.write(out_frame.tobytes())

process2.stdin.close()
process1.wait()
process2.wait()