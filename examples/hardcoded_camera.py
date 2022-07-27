import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

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
        #version 330

        #include "perspective"
        #include "lookat"

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            mat4 projection = perspective(45.0, 16.0 / 9.0, 0.1, 1000.0);
            mat4 view = lookat(vec3(4.0, 3.0, 2.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
            mat4 model = mat4(1.0);
            mat4 mvp = projection * view * model;
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

while window.update():
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
