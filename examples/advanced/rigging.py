import struct
import sys
import zipfile
from itertools import cycle

import assets
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

pack = zipfile.ZipFile(assets.get('humanoid.zip'))

rig = pack.open('humanoid.rig')
bones, frames = struct.unpack('ii', rig.read(8))
pose_frame = cycle([rig.read(bones * 32) for i in range(frames)])

pose_buffer = ctx.buffer(next(pose_frame))
vertex_buffer = ctx.buffer(pack.read('humanoid.mesh'))

uniform_buffer = ctx.buffer(size=64)

ctx.includes['common'] = '''
    layout (std140) uniform Common {
        mat4 mvp;
    };
'''

ctx.includes['bones'] = f'''
    layout (std140) uniform PoseBones {{
        vec4 pose_bone[{bones}];
    }};
'''

ctx.includes['qtransform'] = '''
    vec3 qtransform(vec4 q, vec3 v) {
        return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
    }
'''

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "common"
        #include "bones"
        #include "qtransform"

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in ivec4 in_bone;
        layout (location = 3) in vec4 in_weight;

        vec3 pose_vertex(int bone) {
            return pose_bone[bone * 2 + 0].xyz + qtransform(pose_bone[bone * 2 + 1], in_vert);
        }

        vec3 pose_normal(int bone) {
            return qtransform(pose_bone[bone * 2 + 1], in_norm);
        }

        out vec3 v_norm;

        void main() {
            vec3 v_vert = vec3(0.0);
            v_norm = vec3(0.0);
            v_vert += pose_vertex(in_bone.x) * in_weight.x;
            v_vert += pose_vertex(in_bone.y) * in_weight.y;
            v_vert += pose_vertex(in_bone.z) * in_weight.z;
            v_vert += pose_vertex(in_bone.w) * in_weight.w;
            v_norm += pose_normal(in_bone.x) * in_weight.x;
            v_norm += pose_normal(in_bone.y) * in_weight.y;
            v_norm += pose_normal(in_bone.z) * in_weight.z;
            v_norm += pose_normal(in_bone.w) * in_weight.w;
            v_norm = normalize(v_norm);
            gl_Position = mvp * vec4(v_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(0.0, -3.0, 7.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'PoseBones',
            'binding': 1,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'uniform_buffer',
            'binding': 1,
            'buffer': pose_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 4i 4f', 0, 1, 2, 3),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 4i 4f'),
)

camera = zengl.camera((1.0, -2.0, 1.0), (0.0, 0.0, 0.5), aspect=window_aspect, fov=45.0)
uniform_buffer.write(camera)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now  = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    pose_buffer.write(next(pose_frame))
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
