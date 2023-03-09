import gzip
from colorsys import hls_to_rgb

import numpy as np
import zengl
from chull import make_hull
from objloader import Obj

import assets
from window import Window


def gen_sphere(radius, res=100):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (np.arange(res) / (res - 1.0)) * 2.0
    x = np.cos(phi * np.arange(res)) * np.sqrt(1.0 - y * y)
    z = np.sin(phi * np.arange(res)) * np.sqrt(1.0 - y * y)
    return np.array([x, y, z]).T * radius


window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb')

vertex = ctx.image(window.size, 'rgba32float')
normal = ctx.image(window.size, 'rgba32float')
color = ctx.image(window.size, 'rgba8unorm-srgb')
depth = ctx.image(window.size, 'depth24plus')
color.clear_value = (0.2, 0.2, 0.2, 1.0)
normal.clear_value = (0.0, 0.0, 0.0, 1.0)

model = Obj.frombytes(gzip.decompress(open(assets.get('boxgrid.obj.gz'), 'rb').read())).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

light_vertex_buffer = ctx.buffer(make_hull(gen_sphere(1.0))[0].astype('f4').tobytes())
light_instance_buffer = ctx.buffer(size=4096)

uniform_buffer = ctx.buffer(size=64)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;

        out vec3 v_vertex;
        out vec3 v_normal;

        void main() {
            v_vertex = in_vertex;
            v_normal = in_normal;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec3 v_vertex;
        in vec3 v_normal;

        layout (location = 0) out vec3 out_vertex;
        layout (location = 1) out vec3 out_normal;
        layout (location = 2) out vec3 out_color;

        void main() {
            out_vertex = v_vertex;
            out_normal = v_normal;
            out_color = vec3(1.0, 1.0, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[vertex, normal, color, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

lights = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_position;
        layout (location = 2) in float in_size;
        layout (location = 3) in vec3 in_color;

        out vec3 v_position;
        out float v_size;
        out vec3 v_color;

        void main() {
            v_position = in_position;
            v_size = in_size;
            v_color = in_color;
            gl_Position = mvp * vec4(in_position + in_vertex * v_size, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        uniform sampler2D Vertex;
        uniform sampler2D Normal;
        uniform sampler2D Color;

        in vec3 v_position;
        in float v_size;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            ivec2 t = ivec2(gl_FragCoord.xy);
            vec3 vertex = texelFetch(Vertex, t, 0).xyz;
            vec3 normal = texelFetch(Normal, t, 0).xyz;
            vec3 color = texelFetch(Color, t, 0).rgb;

            float falloff = clamp(v_size - length(v_position - vertex), 0.0, 1.0);
            float lum = dot(normalize(v_position - vertex), normalize(normal)) * falloff;
            out_color = vec4(color * v_color * lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Vertex',
            'binding': 0,
        },
        {
            'name': 'Normal',
            'binding': 1,
        },
        {
            'name': 'Color',
            'binding': 2,
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
            'image': vertex,
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': normal,
        },
        {
            'type': 'sampler',
            'binding': 2,
            'image': color,
        },
    ],
    framebuffer=[image],
    topology='triangles',
    cull_face='back',
    vertex_buffers=[
        *zengl.bind(light_vertex_buffer, '3f', 0),
        *zengl.bind(light_instance_buffer, '3f 1f 3f /i', 1, 2, 3),
    ],
    blend={
        'enable': 1,
        'src_color': 'one',
        'dst_color': 'one',
    },
    vertex_count=light_vertex_buffer.size // zengl.calcsize('3f'),
)

camera = zengl.camera((20.0, 0.0, 10.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

light_color = np.array([hls_to_rgb(x, 0.3, 0.8) for x in np.random.uniform(0.0, 1.0, 49)])
light_instances = np.array([
    np.tile(np.linspace(-9.0, 9.0, 7), 7),
    np.repeat(np.linspace(-9.0, 9.0, 7), 7),
    np.random.uniform(0.2, 1.0, 49),
    np.random.uniform(1.5, 3.0, 49),
    light_color[:, 0],
    light_color[:, 1],
    light_color[:, 2],
]).T.astype('f4')

offset = np.random.uniform(0.0, np.pi, 49)
radius1 = np.random.uniform(1.5, 3.0, 49)
radius2 = np.random.uniform(1.5, 3.0, 49)

while window.update():
    ctx.new_frame()
    light_instances[:, 3] = (radius1 + radius2) / 2.0 + (radius1 - radius2) * np.sin(offset + window.time)
    light_instance_buffer.write(light_instances.tobytes())
    lights.instance_count = 49
    image.clear()
    depth.clear()
    pipeline.render()
    lights.render()
    image.blit()
    ctx.end_frame()
