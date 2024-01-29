import zengl


def test_inspect_pipeline(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    texture = ctx.image((64, 64), 'rgba8unorm')
    uniform_buffer = ctx.buffer(size=64)
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            layout (std140) uniform Common {
                mat4 mvp;
            };

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = mvp * vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            uniform sampler2D Texture;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = texture(Texture, vec2(0.5, 0.5));
            }
        ''',
        layout=[
            {
                'name': 'Common',
                'binding': 0,
            },
            {
                'name': 'Texture',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'uniform_buffer',
                'binding': 0,
                'buffer': uniform_buffer,
            },
            {
                'type': 'sampler',
                'binding': 0,
                'image': texture,
            },
        ],
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

    assert inspect['type'] == 'image'
    assert isinstance(inspect['renderbuffer'], int)

    inspect = zengl.inspect(image.face(0))

    assert inspect['type'] == 'image_face'
    assert isinstance(inspect['framebuffer'], int)


def test_inspect_texture(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    inspect = zengl.inspect(image)

    assert inspect['type'] == 'image'
    assert isinstance(inspect['texture'], int)

    inspect = zengl.inspect(image.face(0))

    assert inspect['type'] == 'image_face'
    assert isinstance(inspect['framebuffer'], int)


def test_inspect_buffer(ctx: zengl.Context):
    buffer = ctx.buffer(size=64)
    inspect = zengl.inspect(buffer)

    assert inspect['type'] == 'buffer'
    assert isinstance(inspect['buffer'], int)
