import gzip
import struct
from math import sin

import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image((512, 512), 'rgba8unorm')
image.clear_value = (0.2, 0.2, 0.2, 1.0)

size = (256, 256)
texture = ctx.image(size, 'rgba8unorm', cubemap=True)
texture.clear_value = (0.1, 0.1, 0.1, 1.0)

temp_depth = ctx.image(size, 'depth24plus')

cube = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        uniform vec2 view;

        mat4 perspective(float fovy, float aspect, float znear, float zfar) {
            float tan_half_fovy = tan(fovy * 0.008726646259971647884618453842);
            return mat4(
                1.0 / (aspect * tan_half_fovy), 0.0, 0.0, 0.0,
                0.0, 1.0 / (tan_half_fovy), 0.0, 0.0,
                0.0, 0.0, -(zfar + znear) / (zfar - znear), -1.0,
                0.0, 0.0, -(2.0 * zfar * znear) / (zfar - znear), 0.0
            );
        }

        mat4 lookat(vec3 eye, vec3 center, vec3 up) {
            vec3 f = normalize(center - eye);
            vec3 s = normalize(cross(f, up));
            vec3 u = cross(s, f);
            return mat4(
                s.x, u.x, -f.x, 0.0,
                s.y, u.y, -f.y, 0.0,
                s.z, u.z, -f.z, 0.0,
                -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0
            );
        }

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

        out vec3 v_vertex;

        void main() {
            v_vertex = vertices[gl_VertexID];
            vec3 eye = vec3(cos(view.x) * cos(view.y), sin(view.x) * cos(view.y), sin(view.y)) * 5.0;
            mat4 mvp = perspective(45.0, 1.0, 0.1, 10.0) * lookat(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core

        uniform samplerCube Texture;

        in vec3 v_vertex;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(texture(Texture, v_vertex).rgb, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
        },
    ],
    uniforms={
        'view': [0.0, 0.0],
    },
    framebuffer=[image],
    topology='triangles',
    cull_face='back',
    vertex_count=36,
)

flat = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        out vec2 v_vertex;

        void main() {
            v_vertex = positions[gl_VertexID];
            gl_Position = vec4(v_vertex, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core

        uniform samplerCube Texture;

        float hash12(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 10.31);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        in vec2 v_vertex;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec2 v = fract(v_vertex * 2.5 + vec2(3.0, 2.5)) * 2.0 - 1.0;
            ivec2 i = ivec2(v_vertex * 2.5 + vec2(3.0, 2.5));
            vec3 t = vec3(0.0, 0.0, 0.0);
            if (i == ivec2(1, 2)) {
                t = vec3(-1.0, v.x, v.y);
            }
            if (i == ivec2(2, 2)) {
                t = vec3(v.x, 1.0, v.y);
            }
            if (i == ivec2(3, 2)) {
                t = vec3(1.0, -v.x, v.y);
            }
            if (i == ivec2(4, 2)) {
                t = vec3(-v.x, -1.0, v.y);
            }
            if (i == ivec2(2, 1)) {
                t = vec3(v.x, v.y, -1.0);
            }
            if (i == ivec2(2, 3)) {
                t = vec3(v.x, -v.y, 1.0);
            }
            out_color = vec4(vec3(hash12(v_vertex)), 1.0);
            if (t != vec3(0.0, 0.0, 0.0)) {
                out_color = vec4(texture(Texture, t).rgb, 1.0);
            }
        }
    ''',
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
        },
    ],
    framebuffer=[image],
    topology='triangles',
    cull_face='back',
    vertex_count=36,
)

model = Obj.frombytes(gzip.decompress(open(assets.get('cubemap-tester.obj.gz'), 'rb').read())).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)
scene_uniform_buffer = ctx.buffer(size=384)


def cubemap_face_pipeline(face):
    image_face = texture.face(layer=face)
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 450 core

            const float znear = 0.1;
            const float zfar = 100.0;
            const float f1 = (zfar + znear) / (zfar - znear);
            const float f2 = -2.0 * zfar * znear / (zfar - znear);

            mat4 mvp[6] = mat4[](
                mat4(0.0, 0.0, f1, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                mat4(0.0, 0.0, -f1, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                mat4(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, f1, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                mat4(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -f1, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, f2, 0.0),
                mat4(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, f1, 1.0, 0.0, 0.0, f2, 0.0),
                mat4(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -f1, -1.0, 0.0, 0.0, f2, 0.0)
            );

            uniform int face;

            layout (location = 0) in vec3 in_vert;
            layout (location = 1) in vec3 in_norm;

            out vec3 v_vert;
            out vec3 v_norm;

            void main() {
                v_vert = in_vert;
                v_norm = in_norm;
                gl_Position = mvp[face] * vec4(v_vert, 1.0);
            }
        ''',
        fragment_shader='''
            #version 450 core

            in vec3 v_vert;
            in vec3 v_norm;

            layout (location = 0) out vec4 out_color;

            void main() {
                vec3 light = vec3(0.0, 0.0, 0.0);
                float lum = dot(normalize(light - v_vert), normalize(v_norm)) * 0.7 + 0.3;
                out_color = vec4(lum, lum, lum, 1.0);
            }
        ''',
        uniforms={
            'face': face,
        },
        framebuffer=[image_face, temp_depth],
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    )
    return image_face, pipeline


scene_pipelines = [cubemap_face_pipeline(i) for i in range(6)]

while window.update():

    for face, pipeline in scene_pipelines:
        face.clear()
        temp_depth.clear()
        pipeline.run()

    t = window.time * 0.5
    cube.uniforms['view'][:] = struct.pack('ff', t, sin(t))

    image.clear()
    if not window.key_down('space'):
        cube.run()
    else:
        flat.run()
    image.blit()
