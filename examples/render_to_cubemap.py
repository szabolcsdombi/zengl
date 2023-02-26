import gzip

import numpy as np
import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

uniform_buffer = ctx.buffer(size=64)

size = (256, 256)
texture = ctx.image(size, 'rgba8unorm', cubemap=True)
texture.clear_value = (0.1, 0.1, 0.1, 1.0)

temp_depth = ctx.image(size, 'depth24plus')

shape = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
        };

        vec3 vertices[36] = vec3[](
            vec3(-1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(-1.0, 1.0, -1.0)
        );

        out vec3 v_text;

        void main() {
            gl_Position = mvp * vec4(vertices[gl_VertexID], 1.0);
            v_text = vertices[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 450 core

        uniform samplerCube Texture;
        in vec3 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(texture(Texture, v_text).rgb, 1.0);
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
    vertex_count=36,
)

model = Obj.frombytes(gzip.decompress(open(assets.get('boxgrid.obj.gz'), 'rb').read())).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)
scene_uniform_buffer = ctx.buffer(size=384)


def cubemap_face_pipeline(face):
    image_face = texture.face(layer=face)
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 450 core

            layout (std140, binding = 0) uniform Common {
                mat4 mvp;
            };

            layout (location = 0) in vec3 in_vert;
            layout (location = 1) in vec3 in_norm;

            out vec3 v_norm;

            void main() {
                gl_Position = mvp * vec4(in_vert - vec3(0.0, 0.0, 4.0), 1.0);
                v_norm = in_norm;
            }
        ''',
        fragment_shader='''
            #version 450 core

            in vec3 v_norm;

            layout (location = 0) out vec4 out_color;

            void main() {
                vec3 light = vec3(4.0, 3.0, 10.0);
                float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
                out_color = vec4(lum, lum, lum, 1.0);
            }
        ''',
        resources=[
            {
                'type': 'uniform_buffer',
                'binding': 0,
                'buffer': scene_uniform_buffer,
                'offset': 64 * face,
            },
        ],
        framebuffer=[image_face, temp_depth],
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    )
    return image_face, pipeline


scene_pipelines = [cubemap_face_pipeline(i) for i in range(6)]

scene_uniform_buffer.write(b''.join([
    zengl.camera((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), fov=90.0),
    zengl.camera((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), fov=90.0),
    zengl.camera((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), fov=90.0),
    zengl.camera((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), fov=90.0),
    zengl.camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0), fov=90.0),
    zengl.camera((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), fov=90.0),
]))

while window.update():
    ctx.new_frame()

    for face, pipeline in scene_pipelines:
        face.clear()
        temp_depth.clear()
        pipeline.render()

    t = window.time * 0.5
    eye = (np.cos(t) * 5.0, np.sin(t) * 5.0, np.sin(t * 0.7) * 2.0)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    shape.render()
    image.blit()
    ctx.end_frame()
