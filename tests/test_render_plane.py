import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    depth = ctx.image((64, 64), 'depth24plus')
    texture = ctx.image((16, 16), 'rgba8unorm', np.full((16, 16, 4), (64, 64, 255, 255), 'u1'))
    uniform_buffer = ctx.buffer(size=64)
    vertex_buffer = ctx.buffer(
        np.array(
            [
                [-1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            'f4',
        ).T
    )
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            layout (std140) uniform Common {
                mat4 mvp;
            };

            layout (location = 0) in vec3 in_vert;
            layout (location = 1) in vec2 in_text;

            out vec2 v_text;

            void main() {
                v_text = in_text;
                gl_Position = mvp * vec4(in_vert, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            uniform sampler2D Texture;

            in vec2 v_text;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = texture(Texture, v_text);
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
        framebuffer=[image, depth],
        topology='triangle_strip',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 2f', 0, 1),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 2f'),
    )

    camera = zengl.camera((0.4, 0.3, 0.2), (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)

    ctx.new_frame()
    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(pixels[32, 32], [64, 64, 255, 255])
