import io
import json
import struct
import zipfile

import numpy as np
import vmath
import zengl

import assets
from window import Window


def read_gltf(f):
    sig, ver, _ = struct.unpack('III', f.read(12))
    assert sig == 0x46546c67 and ver == 2
    size, code = struct.unpack('II', f.read(8))
    assert code == 0x4e4f534a
    data = json.loads(f.read(size))

    buffers = []
    while header := f.read(8):
        size, code = struct.unpack('II', header)
        assert code == 0x4e4942
        buffers.append(f.read(size))

    return data, buffers


class Loader:
    def __init__(self):
        self.data = None
        self.buffers = None
        self.output = io.BytesIO()
        self.idx = 0

    def load(self, data):
        self.data, self.buffers = read_gltf(io.BytesIO(data))
        self.export()
        self.idx = self.idx + 1

    def get_color(self, material):
        r, g, b, _ = self.data['materials'][material]['pbrMetallicRoughness']['baseColorFactor']
        return round(r * 255), round(g * 255), round(b * 255)

    def get_chunk(self, accessor, size):
        acc = self.data['accessors'][accessor]
        view = self.data['bufferViews'][acc['bufferView']]
        chunk = self.buffers[view['buffer']][view['byteOffset']:view['byteOffset'] + view['byteLength']]
        return chunk[acc['byteOffset']:acc['byteOffset'] + acc['count'] * size]

    def get_mesh(self, mesh):
        vertices = []
        for primitive in self.data['meshes'][mesh]['primitives']:
            color = self.get_color(primitive['material'])
            vert = self.get_chunk(primitive['attributes']['POSITION'], 12)
            norm = self.get_chunk(primitive['attributes']['NORMAL'], 12)
            idx = self.get_chunk(primitive['indices'], 4)
            for i in range(0, len(idx), 4):
                k = int.from_bytes(idx[i:i + 4], 'little')
                vertices.append((
                    struct.unpack('fff', vert[k * 12:k * 12 + 12]),
                    struct.unpack('fff', norm[k * 12:k * 12 + 12]),
                    color,
                ))
        return vertices

    def visit(self, node, parent_frame):
        info = self.data['nodes'][node]
        translation = vmath.vec(*info['translation'])
        rotation = vmath.quat(*info['rotation'])
        scale = vmath.vec(*info['scale'])
        normal_scale = vmath.vec(1.0, 1.0, 1.0) / scale
        frame = parent_frame * vmath.mat(translation, rotation, scale)
        normal_frame = parent_frame * vmath.mat(vmath.vec(0.0, 0.0, 0.0), rotation, normal_scale)
        if 'mesh' in info:
            mesh = self.get_mesh(info['mesh'])
            for vert, norm, color in mesh:
                vx, vy, vz = frame * vmath.vec(*vert)
                nx, ny, nz = normal_frame * vmath.vec(*norm)
                self.output.write(struct.pack('fff', vx, vy, vz))
                self.output.write(struct.pack('fff', nx, ny, nz))
                self.output.write(struct.pack('BBBB', *color, 255))
                self.output.write(struct.pack('i', self.idx))
        for child in info.get('children', []):
            self.visit(child, frame)

    def export(self):
        frame = vmath.mat(vmath.vec(0.0, 0.0, 0.0), vmath.rotate_x(vmath.pi / 2), vmath.vec(1.0, 1.0, 1.0))
        for node in self.data['scenes'][self.data['scene']]['nodes']:
            self.visit(node, frame)


loader = Loader()

pack = zipfile.ZipFile(assets.get('furniturekit_updated.zip'))  # https://www.kenney.nl/assets/furniture-kit
for name in ['chairCushion', 'tableCloth', 'ceilingFan']:
    loader.load(pack.read(f'Models/GLTF format/{name}.glb'))

pack = zipfile.ZipFile(assets.get('foodKit_v1.2.zip'))  # https://www.kenney.nl/assets/food-kit
for name in ['plate', 'utensilFork', 'utensilSpoon', 'utensilKnife']:
    loader.load(pack.read(f'Models/GLTF format/{name}.glb'))

names = [x for x in pack.namelist() if x.startswith('Models/GLTF format') and x.endswith('.glb')]
for name in names:
    loader.load(pack.read(name))

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

vertex_buffer = ctx.buffer(loader.output.getvalue())
uniform_buffer = ctx.buffer(size=64)
bone_buffer = ctx.buffer(size=256 * 32)

ctx.includes['qtransform'] = '''
    vec3 qtransform(vec4 q, vec3 v) {
        return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
    }
'''

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330

        #include "qtransform"

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (std140) uniform Bones {
            vec4 bone[256 * 2];
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec3 in_color;
        layout (location = 3) in int in_bone;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec3 v_color;

        void main() {
            float scale = bone[in_bone * 2 + 0].w;
            vec3 position = bone[in_bone * 2 + 0].xyz;
            vec4 rotation = bone[in_bone * 2 + 1];
            v_vert = position + qtransform(rotation, in_vert * scale);
            v_norm = normalize(qtransform(rotation, in_norm));
            gl_Position = mvp * vec4(v_vert, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_norm;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(1.0, -4.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(v_color * lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Bones',
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
            'buffer': bone_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 4nu1 1i', 0, 1, 2, 3),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 4nu1 1i'),
)

camera = zengl.camera((1.0, -4.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

bones = np.full((256, 8), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 'f4')

bones[0, :] = [-0.1, 0.12, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
bones[1, :] = [-0.42, -0.225, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
bones[2, :] = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0]
bones[3, :] = [0.0, 0.0, 0.325, 0.245, 0.0, 0.0, 0.0, 1.0]
bones[4, :] = [0.14, 0.0, 0.335, 0.45, 0.0, 0.0, -0.707, 0.707]
bones[5, :] = [-0.19, 0.0, 0.335, 0.45, 0.0, 0.0, -0.707, 0.707]
bones[6, :] = [-0.14, 0.0, 0.335, 0.4, 0.0, 0.0, -0.707, 0.707]

bones[147, 3] = 0.5
bones[149:155, 3] = 0.5
bones[184:186, 3] = 0.5

for i in range(7, loader.idx):
    bones[i, 4:8] = vmath.random_rotation()

offset = np.random.uniform(0.0, np.pi * 2.0, 256)
speed = np.random.uniform(0.5, 0.7, 256)
radius = np.random.uniform(2.0, 3.0, 256)
vertical_offset = np.random.uniform(0.0, np.pi * 2.0, 256)
vertical_speed = np.random.uniform(0.2, 0.4, 256)
vertical_radius = np.random.uniform(0.1, 0.2, 256)
rotation_speed = np.random.uniform(0.01, 0.02, 256)
rotation_axis = [vmath.random_axis() for _ in range(256)]

while window.update():
    bones[2, 4:8] = vmath.rotate_z(0.3) * vmath.quat(bones[2, 4:8])
    s = np.sin(offset + window.time * speed) * radius
    c = np.cos(offset + window.time * speed) * radius
    v = np.sin(vertical_offset + window.time * vertical_speed) * vertical_radius + 0.5
    for i in range(7, loader.idx):
        bones[i, 0:3] = c[i], s[i], v[i]
        bones[i, 4:8] = vmath.rotate(rotation_axis[i], rotation_speed[i]) * vmath.quat(bones[i, 4:8])
    bone_buffer.write(bones)
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
