import zengl
from PIL import Image

from window import Window

window = Window(1280, 640)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb', samples=16)
depth = ctx.image(window.size, 'depth24plus', samples=16)
image.clear_value = (0.0, 0.0, 0.0, 0.0)

ctx.includes['perspective'] = '''
    mat4 perspective(float fovy, float aspect, float znear, float zfar) {
        float tan_half_fovy = tan(fovy * 0.008726646259971647884618453842);
        return mat4(
            1.0 / (aspect * tan_half_fovy), 0.0, 0.0, 0.0,
            0.0, 1.0 / (tan_half_fovy), 0.0, 0.0,
            0.0, 0.0, -(zfar + znear) / (zfar - znear), -1.0,
            0.0, 0.0, -(2.0 * zfar * znear) / (zfar - znear), 0.0
        );
    }
'''

ctx.includes['lookat'] = '''
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
'''

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        #include "perspective"
        #include "lookat"

        const float pi = 3.1415926535897932384626;

        vec3 positions[4] = vec3[](
            vec3(0.0, 0.0, 0.5),
            vec3(0.0, 1.0, -0.5),
            vec3(1.0, 0.0, 0.5),
            vec3(1.0, 1.0, -0.5)
        );

        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        out vec3 v_vertex;
        out vec3 v_color;

        void main() {
            mat4 projection = perspective(45.0, 2.0, 0.1, 1000.0);
            mat4 view = lookat(vec3(0.0, 3.75, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
            mat4 mvp = projection * view;

            v_vertex = positions[gl_VertexID];

            float r = pi - float(gl_InstanceID) * pi * 2.0 / 7.0;
            mat3 rotation = mat3(cos(r), 0.0, sin(r), 0.0, 1.0, 0.0, -sin(r), 0.0, cos(r));

            gl_Position = mvp * vec4(rotation * v_vertex, 1.0);
            v_color = hsv2rgb(vec3(float(gl_InstanceID) / 10.0, 1.0, 0.5));
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_vertex;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            float u = smoothstep(0.48, 0.47, abs(v_vertex.x - 0.5));
            float v = smoothstep(0.48, 0.47, abs(v_vertex.y - 0.5));
            out_color = vec4(v_color * (u * v), 1.0);
        }
    ''',
    framebuffer=[image, depth],
    topology='triangle_strip',
    vertex_count=4,
    instance_count=7,
)

depth.clear()
image.clear()
pipeline.render()

temp = ctx.image(image.size, 'rgba8unorm-srgb')
image.blit(temp)
Image.frombuffer('RGBA', temp.size, temp.read(), 'raw', 'RGBA', 0, -1).save('logo.png')

while window.update():
    image.blit()
