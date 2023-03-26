import zengl


def test_inspect_pipeline(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    pipeline = ctx.pipeline(
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
                out_color = vec4(0.0, 0.0, 1.0, 1.0);
            }
        ''',
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )

    inspect = zengl.inspect(pipeline)

    assert inspect['type'] == 'pipeline'
    assert isinstance(inspect['framebuffer'], int)
    assert isinstance(inspect['program'], int)
    assert isinstance(inspect['vertex_array'], int)
    assert 'interface' in inspect
    assert 'resources' in inspect


def test_inspect_renderbuffer(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm', samples=4)
    inspect = zengl.inspect(image)

    assert inspect['type'] == 'renderbuffer'
    assert isinstance(inspect['renderbuffer'], int)
    assert isinstance(inspect['framebuffer'], int)


def test_inspect_texture(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    inspect = zengl.inspect(image)

    assert inspect['type'] == 'texture'
    assert isinstance(inspect['texture'], int)
    assert isinstance(inspect['framebuffer'], int)


def test_inspect_buffer(ctx: zengl.Context):
    buffer = ctx.buffer(size=64)
    inspect = zengl.inspect(buffer)

    assert inspect['type'] == 'buffer'
    assert isinstance(inspect['buffer'], int)
