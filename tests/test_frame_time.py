import zengl


def test_frame_time(ctx: zengl.Context):
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
                gl_Position = vec4(positions[gl_VertexID] + 0.5, 0.0, 1.0);
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

    ctx.new_frame()
    image.clear()
    pipeline.render()
    ctx.end_frame()

    assert ctx.frame_time == 0.0

    ctx.new_frame(frame_time=True)
    image.clear()
    pipeline.render()
    ctx.end_frame(sync=True)

    assert ctx.frame_time > 0.0

    ctx.new_frame()
    image.clear()
    pipeline.render()
    ctx.end_frame()

    assert ctx.frame_time == 0.0
