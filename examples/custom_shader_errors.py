import io
import re
from collections import defaultdict

import _zengl
import colorama
import zengl

colorama.init()


def compile_error(shader: bytes, shader_type: int, log: bytes):
    log = log.rstrip(b'\x00').strip().decode().splitlines()
    errors = defaultdict(list)
    for line in log:
        match = re.search(r'\d+:(\d+):', line)
        if not match:
            errors[-1].append(line)
            continue
        line_number = int(match.group(1))
        errors[line_number].append(line[match.span()[1]:].strip())
    res = io.StringIO()
    for i, line in enumerate(shader.decode().split('\n'), 1):
        print(f'{i:4d} | {line}', file=res)
        for error in errors[i]:
            print(f'     | \x1b[31m{error}\x1b[m', file=res)
    for error in errors[-1]:
        print(f'     | \x1b[31m{error}\x1b[m', file=res)
    raise ValueError(f'GLSL Compile Error:\n\n{res.getvalue()}')


def linker_error(vertex_shader: bytes, fragment_shader: bytes, log: bytes):
    res = io.StringIO()
    print('Vertex Shader', file=res)
    print('=============', file=res)
    for i, line in enumerate(vertex_shader.decode().split('\n'), 1):
        print(f'{i:4d} | {line}', file=res)
    print('', file=res)
    print('Fragment Shader', file=res)
    print('===============', file=res)
    for i, line in enumerate(fragment_shader.decode().split('\n'), 1):
        print(f'{i:4d} | {line}', file=res)
    print('', file=res)
    error = log.rstrip(b'\x00').decode().strip()
    print(f'\x1b[31m{error}\x1b[m', file=res)
    raise ValueError(f'GLSL Linker Error:\n\n{res.getvalue()}')


_zengl.compile_error = compile_error
_zengl.linker_error = linker_error

ctx = zengl.context(zengl.loader(headless=True))
image = ctx.image((256, 256), 'rgba8unorm')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec2 positions[3] = vec2[4](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)
