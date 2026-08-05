import numpy as np
import zengl
from OpenGL import GL

import utils


def make_pipeline(ctx, binding_1, binding_2):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            uniform sampler2D Texture_1;

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID] + texture(Texture_1, vec2(0.5, 0.5)).rg, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            uniform sampler2D Texture_2;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = texture(Texture_2, vec2(0.5, 0.5));
            }
        ''',
        layout=[
            {
                'name': 'Texture_1',
                'binding': binding_1,
            },
            {
                'name': 'Texture_2',
                'binding': binding_2,
            },
        ],
        resources=[
            {
                'type': 'sampler',
                'binding': binding_1,
                'image': ctx.image((16, 16), 'rg32float'),
            },
            {
                'type': 'sampler',
                'binding': binding_2,
                'image': ctx.image((16, 16), 'rgba8unorm'),
            },
        ],
        framebuffer=[ctx.image((64, 64), 'rgba8unorm')],
        topology='triangles',
        vertex_count=3,
    )


def read_bindings(pipeline):
    program = zengl.inspect(pipeline)['program']
    uniforms = sorted(zengl.inspect(pipeline)['interface'][1], key=lambda x: x['name'])
    bindings = []
    for uniform in uniforms:
        value = np.zeros(1, 'i4')
        GL.glGetUniformiv(program, uniform['location'], value)
        bindings.append(int(value[0]))
    return bindings


def test(ctx: zengl.Context):
    pipeline = make_pipeline(ctx, 1, 3)
    other = make_pipeline(ctx, 5, 7)

    assert zengl.inspect(pipeline)['program'] != zengl.inspect(other)['program']

    utils.clear_gl_error()
    cached = make_pipeline(ctx, 1, 3)
    utils.assert_gl_error('building a pipeline with a cached program')

    assert zengl.inspect(cached)['program'] == zengl.inspect(pipeline)['program']
    assert read_bindings(pipeline) == [1, 3]
    assert read_bindings(other) == [5, 7]
