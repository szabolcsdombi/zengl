import math

import zengl
from objloader import Obj
from PIL import Image

from window import Window

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

img = Image.open('examples/data/forest.jpg').convert('RGBA')  # https://polyhaven.com/a/phalzer_forest_01
texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())

model = Obj.open('examples/data/blob.obj').pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=80)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 eye;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_vert;
        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 eye;
        };

        uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        float atan2(float y, float x) {
            return x == 0.0 ? sign(y) * 3.1415 / 2 : atan(y, x);
        }

        void main() {
            vec3 ray = reflect(normalize(v_vert - eye), normalize(v_norm));
            vec2 tex = vec2(atan2(ray.y, ray.x) / 3.1415, -ray.z);
            float lum = dot(normalize(eye - v_vert), normalize(v_norm)) * 0.3 + 0.7;
            vec3 color = texture(Texture, tex).rgb;
            out_color = vec4(color * lum, 1.0);
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
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    x, y = math.sin(window.time * 0.5) * 5.0, math.cos(window.time * 0.5) * 5.0
    camera = zengl.camera((x, y, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera)
    uniform_buffer.write(zengl.pack(x, y), offset=64)

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
