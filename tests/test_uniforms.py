import zengl
import numpy as np


def make_pipeline(ctx, image, gltype, color):
    return ctx.pipeline(
        vertex_shader='''
            #version 300 es
            precision highp float;
            vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
            void main() {
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 300 es
            precision highp float;
            precision highp int;
            #include "TYPE"
            uniform TYPE color;
            layout (location = 0) out TYPE out_color;
            void main() {
                out_color = color;
            }
        ''',
        includes={
            'TYPE': f'#define TYPE {gltype}',
        },
        uniforms={
            'color': color,
        },
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )


def test_uniform_vec4(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba32float', bytearray(256))

    pipeline = make_pipeline(ctx, img, 'vec4', [0.0, 0.25, 0.5, 1.0])
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'f4')
    np.testing.assert_array_almost_equal(pixel, [0.0, 0.25, 0.5, 1.0], decimal=3)

    pipeline = make_pipeline(ctx, img, 'vec4', [0.33, 0.33, 0.33, 1.0])
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'f4')
    np.testing.assert_array_almost_equal(pixel, [0.33, 0.33, 0.33, 1.0], decimal=3)


def test_uniform_ivec4(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba32sint', bytearray(256))

    pipeline = make_pipeline(ctx, img, 'ivec4', [0, 100, -1024, 0x11223344])
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'i4')
    np.testing.assert_array_equal(pixel, [0, 100, -1024, 0x11223344])

    pipeline = make_pipeline(ctx, img, 'ivec4', [-1, -1, 100000, 100000])
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'i4')
    np.testing.assert_array_equal(pixel, [-1, -1, 100000, 100000])


def test_uniform_uvec4(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba32uint', bytearray(256))

    pipeline = make_pipeline(ctx, img, 'uvec4', [0, 100, 1024, 0xFF000000])
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'u4')
    np.testing.assert_array_equal(pixel, [0, 100, 1024, 0xFF000000])

    pipeline = make_pipeline(ctx, img, 'uvec4', [255, 127, 63, 0])
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'u4')
    np.testing.assert_array_equal(pixel, [255, 127, 63, 0])


def test_uniform_mat4(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba32float', bytearray(256))

    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 300 es
            precision highp float;
            vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
            void main() {
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 300 es
            precision highp float;
            uniform mat4 kernel;
            uniform vec4 color;
            layout (location = 0) out vec4 out_color;
            void main() {
                out_color = kernel * color;
            }
        ''',
        uniforms={
            'kernel': [0.0] * 16,
            'color': [1.0, 2.0, 3.0, 4.0],
        },
        framebuffer=[img],
        topology='triangles',
        vertex_count=3,
    )

    kernel = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    pipeline.uniforms['kernel'][:] = np.array(kernel, 'f4').tobytes()
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'f4')
    np.testing.assert_array_almost_equal(pixel, [1.0, 2.0, 3.0, 4.0], decimal=3)

    kernel = [
        [-2.053, 0.115, 0.683, -0.524],
        [-1.753, -1.090, -0.333, -0.641],
        [1.937, 0.477, -3.711, -0.760],
        [-0.865, 1.234, 2.012, 0.132],
    ]

    pipeline.uniforms['kernel'][:] = np.array(kernel, 'f4').tobytes()
    pipeline.render()
    pixel = np.frombuffer(img.read((1, 1)), 'f4')
    np.testing.assert_array_almost_equal(pixel, [-3.208, 4.302, -3.068, -3.558], decimal=3)
