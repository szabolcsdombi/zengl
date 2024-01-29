import zengl


def make_pipeline(ctx, framebuffer, viewport, color):
    return ctx.pipeline(
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

            #include "color"

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(color, 1.0);
            }
        ''',
        framebuffer=framebuffer,
        viewport=viewport,
        topology='triangles',
        vertex_count=3,
        includes={
            'color': f'vec3 color = vec3({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f});',
        },
    )


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    depth = ctx.image((64, 64), 'depth24plus')
    depth.clear_value = 0.5

    pipeline_1 = make_pipeline(ctx, [image, depth], (0, 0, 32, 32), (0.0, 0.0, 1.0))
    pipeline_2 = make_pipeline(ctx, [image, depth], (32, 32, 32, 32), (0.0, 0.0, 1.0))
    pipeline_3 = make_pipeline(ctx, [image, depth], (0, 32, 32, 32), (0.0, 1.0, 0.0))
    pipeline_4 = make_pipeline(ctx, [image], (32, 0, 32, 32), (0.0, 0.0, 1.0))

    info_1 = zengl.inspect(pipeline_1)
    info_2 = zengl.inspect(pipeline_2)
    info_3 = zengl.inspect(pipeline_3)
    info_4 = zengl.inspect(pipeline_4)

    assert info_1['vertex_array'] == info_2['vertex_array'] == info_3['vertex_array'] == info_4['vertex_array']

    assert info_1['framebuffer'] == info_2['framebuffer'] == info_3['framebuffer']
    assert info_1['framebuffer'] != info_4['framebuffer']

    assert info_1['program'] == info_2['program'] == info_4['program']
    assert info_1['program'] != info_3['program']
