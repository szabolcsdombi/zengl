import numpy as np
import zengl


def make_pipeline(ctx, framebuffer, viewport, depth_function):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            vec3 positions[3] = vec3[](
                vec3(0.1, 0.0, 0.0),
                vec3(-0.05, 0.086, 0.0),
                vec3(-0.05, -0.086, 0.0)
            );

            vec3 colors[3] = vec3[](
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0)
            );

            out vec3 v_color;

            void main() {
                v_color = colors[gl_InstanceID];
                gl_Position = vec4(positions[gl_VertexID] + vec3(0.0, 0.0, float(gl_InstanceID) - 1.0), 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            in vec3 v_color;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(v_color, 1.0);
            }
        ''',
        framebuffer=framebuffer,
        depth={
            'func': depth_function,
            'write': True,
        },
        viewport=viewport,
        topology='triangles',
        vertex_count=3,
        instance_count=3,
    )


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    depth = ctx.image((64, 64), 'depth24plus')
    depth.clear_value = 0.5

    pipeline_1 = make_pipeline(ctx, [image, depth], (0, 0, 32, 32), 'less')
    pipeline_2 = make_pipeline(ctx, [image, depth], (32, 32, 32, 32), 'always')
    pipeline_3 = make_pipeline(ctx, [image, depth], (0, 32, 32, 32), 'greater')
    pipeline_4 = make_pipeline(ctx, [image, depth], (32, 0, 32, 32), 'equal')

    ctx.new_frame()
    image.clear()
    depth.clear()
    pipeline_1.render()
    pipeline_2.render()
    pipeline_3.render()
    pipeline_4.render()
    ctx.end_frame()
    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
    )
