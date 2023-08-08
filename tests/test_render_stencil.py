import numpy as np
import zengl


def make_pipeline(ctx, framebuffer, color, stencil):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID] + 0.5, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            uniform vec3 color;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(color, 1.0);
            }
        ''',
        uniforms={
            'color': color,
        },
        stencil={
            'both': stencil,
        },
        depth={
            'func': 'always',
            'write': False,
        },
        framebuffer=framebuffer,
        topology='triangles',
        vertex_count=3,
    )


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    stencil = ctx.image((64, 64), 'depth24plus-stencil8')
    stencil.clear_value = 1.0, 0

    pipeline_1 = make_pipeline(
        ctx,
        [image, stencil],
        (1.0, 0.0, 0.0),
        {
            'fail_op': 'replace',
            'pass_op': 'replace',
            'depth_fail_op': 'replace',
            'compare_op': 'always',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
    )

    pipeline_2 = make_pipeline(
        ctx,
        [image, stencil],
        (0.0, 1.0, 0.0),
        {
            'fail_op': 'keep',
            'pass_op': 'keep',
            'depth_fail_op': 'keep',
            'compare_op': 'equal',
            'compare_mask': 0xFF,
            'write_mask': 0xFF,
            'reference': 20,
        },
    )

    pipeline_3 = make_pipeline(
        ctx,
        [image, stencil],
        (0.0, 0.0, 1.0),
        {
            'fail_op': 'keep',
            'pass_op': 'keep',
            'depth_fail_op': 'keep',
            'compare_op': 'equal',
            'compare_mask': 0xFF,
            'write_mask': 0xFF,
            'reference': 1,
        },
    )

    ctx.new_frame()
    image.clear()
    stencil.clear()
    pipeline_1.render()
    ctx.end_frame()

    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 0, 0, 255],
        ],
    )

    ctx.new_frame()
    pipeline_2.render()
    ctx.end_frame()

    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 0, 0, 255],
        ],
    )

    ctx.new_frame()
    pipeline_3.render()
    ctx.end_frame()

    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 255, 255],
        ],
    )
