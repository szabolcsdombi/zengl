import pytest
import zengl


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
