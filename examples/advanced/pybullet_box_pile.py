import struct
import sys

import assets
import numpy as np
import pybullet as pb
import pygame
import zengl
import zengl_extras
from objloader import Obj
from PIL import Image

pb.connect(pb.DIRECT)

pb.setGravity(0.0, 0.0, -10.0)
pb.setRealTimeSimulation(0)
pb.setTimeStep(1.0 / 60.0)

plane_shape = pb.createCollisionShape(pb.GEOM_PLANE, planeNormal=(0.0, 0.0, 1.0))
box_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=(0.5, 0.5, 0.5))

ground = pb.createMultiBody(baseMass=0.0, basePosition=(0.0, 0.0, 0.0), baseCollisionShapeIndex=plane_shape)

crates = 50
bullet_crates = []
for i in range(crates):
    x, y = np.random.uniform(-1.0, 1.0, 2)
    obj = pb.createMultiBody(baseMass=1.0, basePosition=(x, y, i * 1.2 + 1), baseCollisionShapeIndex=box_shape)
    bullet_crates.append(obj)

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

instance_array = np.zeros((crates, 8)).astype('f4')
instance_buffer = ctx.buffer(instance_array.tobytes())

img = Image.open(assets.get('crate.png')).convert('RGBA')
texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())
texture.mipmaps()

uniform_buffer = ctx.buffer(size=80)

ctx.includes['qtransform'] = '''
    vec3 qtransform(vec4 q, vec3 v) {
        return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
    }
'''

crate = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "qtransform"

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec2 in_text;
        layout (location = 3) in vec3 in_pos;
        layout (location = 4) in vec4 in_quat;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec2 v_text;

        void main() {
            v_vert = in_pos + qtransform(in_quat, in_vert);
            gl_Position = mvp * vec4(v_vert, 1.0);
            v_norm = qtransform(in_quat, in_norm);
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
            'min_filter': 'linear_mipmap_linear',
            'mag_filter': 'linear',
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        *zengl.bind(instance_buffer, '3f 1f 4f /i', 3, -1, 4),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
    instance_count=crates,
)

camera = zengl.camera((8.0, 6.0, 4.0), (0.0, 0.0, 0.5), aspect=window_aspect, fov=45.0)

uniform_buffer.write(camera)
uniform_buffer.write(struct.pack('3f4x', 8.0, 6.0, 14.0), offset=64)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pb.stepSimulation()

    ctx.new_frame()
    for i, obj in enumerate(bullet_crates):
        pos, rot = pb.getBasePositionAndOrientation(obj)
        instance_array[i, :3] = pos
        instance_array[i, 4:] = rot

    instance_buffer.write(instance_array)

    image.clear()
    depth.clear()
    crate.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
