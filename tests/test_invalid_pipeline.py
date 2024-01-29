import pytest
import zengl

simple_vertex_shader = '''
    #version 300 es
    precision highp float;
    void main() {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    }
'''

simple_fragment_shader = '''
    #version 300 es
    precision highp float;
    layout (location = 0) out vec4 out_color;
    void main() {
        out_color = vec4(0.0);
    }
'''


def test_missing_vertex_shader(ctx: zengl.Context):
    image = ctx.image((4, 4), 'rgba8unorm')
    with pytest.raises(TypeError):
        ctx.pipeline(
            fragment_shader='''
                #version 330 core
                void main() {
                }
            ''',
            framebuffer=[image],
        )


def test_missing_fragment_shader(ctx: zengl.Context):
    image = ctx.image((4, 4), 'rgba8unorm')
    with pytest.raises(TypeError):
        ctx.pipeline(
            vertex_shader='''
                #version 330 core
                void main() {
                    gl_Position = vec4(1.0);
                }
            ''',
            framebuffer=[image],
        )


def test_missing_framebuffer(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.pipeline(
            vertex_shader='''
                #version 330 core
                void main() {
                    gl_Position = vec4(1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core
                void main() {
                }
            ''',
        )


def test_unbound_attribute(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;
                layout (location = 0) in vec2 in_vertex;
                void main() {
                    gl_Position = vec4(in_vertex, 0.0, 1.0);
                }
            ''',
            fragment_shader=simple_fragment_shader,
            framebuffer=None,
            viewport=(0, 0, 4, 4),
        )


def test_invalid_vertex_format(ctx: zengl.Context):
    vbo = ctx.buffer(size=256)

    with pytest.raises(ValueError):
        ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;
                layout (location = 0) in vec2 in_vertex;
                void main() {
                    gl_Position = vec4(in_vertex, 0.0, 1.0);
                }
            ''',
            fragment_shader=simple_fragment_shader,
            framebuffer=None,
            viewport=(0, 0, 4, 4),
            vertex_buffers=[
                {
                    'location': 0,
                    'buffer': vbo,
                    'format': 'float64x2',
                    'offset': 0,
                    'step': 'vertex',
                    'stride': 8,
                },
            ],
        )


def test_invalid_topology(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.pipeline(
            vertex_shader=simple_vertex_shader,
            fragment_shader=simple_fragment_shader,
            framebuffer=None,
            viewport=(0, 0, 4, 4),
            topology='bad',
        )


def test_invalid_render_data(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.pipeline(
            vertex_shader=simple_vertex_shader,
            fragment_shader=simple_fragment_shader,
            framebuffer=None,
            viewport=(0, 0, 4, 4),
            render_data='bad',
        )


def test_invalid_viewport_data(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.pipeline(
            vertex_shader=simple_vertex_shader,
            fragment_shader=simple_fragment_shader,
            framebuffer=None,
            viewport=(0, 0, 4, 4),
            viewport_data=memoryview(bytearray(1000)),
        )


def test_invalid_cull_face(ctx: zengl.Context):
    with pytest.raises(KeyError):
        ctx.pipeline(
            vertex_shader=simple_vertex_shader,
            fragment_shader=simple_fragment_shader,
            framebuffer=None,
            viewport=(0, 0, 4, 4),
            cull_face='bad',
        )
