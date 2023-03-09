import math
from colorsys import hls_to_rgb

import zengl
from objloader import Obj
from OpenGL import GL

import assets
from window import Window

'''
    This is the NOT recommanded way to do it.
    Check out uniform buffers and per-instance attributes.

    TODO: add reference to examples using:
        - per object bound chunk of uniform buffer from a single large uniform buffer
        - per object bound per instance vertex attributes from a single larger vertex buffer

    Getting simple uniforms working:

    - Extract the OpenGL Program Object with zengl.inspect(pipeline)['program']
    - Use glGetUniformLocation and glUniform* as usual
    - Beware of multiple pipelines may have the same program object
'''

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        uniform mat4 mvp;

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330 core

        uniform vec3 color;

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(color * lum, 1.0);
        }
    ''',
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

program = zengl.inspect(pipeline)['program']
mvp = GL.glGetUniformLocation(program, 'mvp')
color = GL.glGetUniformLocation(program, 'color')

while window.update():
    ctx.new_frame()
    x, y = math.sin(window.time * 0.5) * 3.0, math.cos(window.time * 0.5) * 3.0
    camera = zengl.camera((x, y, 1.5), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
    GL.glProgramUniformMatrix4fv(program, mvp, 1, False, camera)
    red, green, blue = hls_to_rgb(window.time * 0.1, 0.5, 0.5)
    GL.glProgramUniform3f(program, color, red, green, blue)

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()
