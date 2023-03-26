import pytest
import zengl


def test_vertex_shader_error(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')

    with pytest.raises(ValueError, match='Vertex Shader Error'):
        ctx.pipeline(
            vertex_shader='''
                #version 330 core

                void main() {
                    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(0.0, 0.0, 0.0, 1.0);
                }
            ''',
            framebuffer=[image],
            topology='triangles',
            vertex_count=3,
        )


def test_fragment_shader_error(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')

    with pytest.raises(ValueError, match='Fragment Shader Error'):
        ctx.pipeline(
            vertex_shader='''
                #version 330 core

                vec2 positions[3] = vec2[](
                    vec2(0.1, 0.0),
                    vec2(-0.05, 0.086),
                    vec2(-0.05, -0.086)
                );

                void main() {
                    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec3(0.0, 0.0, 0.0);
                }
            ''',
            framebuffer=[image],
            topology='triangles',
            vertex_count=3,
        )


def test_linker_error(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')

    with pytest.raises(ValueError, match='Linker Error'):
        ctx.pipeline(
            vertex_shader='''
                #version 330 core

                vec2 positions[3] = vec2[](
                    vec2(0.1, 0.0),
                    vec2(-0.05, 0.086),
                    vec2(-0.05, -0.086)
                );

                out vec3 v_color;

                void main() {
                    v_color = vec3(0.0, 0.0, 1.0);
                    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                in vec4 v_color;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = v_color;
                }
            ''',
            framebuffer=[image],
            topology='triangles',
            vertex_count=3,
        )
