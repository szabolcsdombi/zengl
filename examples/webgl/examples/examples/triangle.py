import struct

import zengl
import webgl

window = webgl.window()

ctx = zengl.context(window)
print(ctx.info, ctx.limits)

image = ctx.image(window.size, 'rgba8unorm', texture=False)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        out vec3 v_color;

        vec2 positions[3] = vec2[](
            vec2(1.0, 0.0),
            vec2(-0.5, -0.86),
            vec2(-0.5, 0.86)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        uniform float time;
        uniform vec2 scale;

        void main() {
            mat2 rot = mat2(cos(time), sin(time), -sin(time), cos(time));
            gl_Position = vec4(rot * positions[gl_VertexID] * scale, 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
        }
    ''',
    framebuffer=[image],
    uniforms={
        'time': 0.0,
        'scale': (0.8 / window.aspect, 0.8),
    },
    topology='triangles',
    vertex_count=3,
)

print(zengl.inspect(pipeline))


def render():
    window.update()
    ctx.new_frame()
    image.clear()
    pipeline.uniforms['time'][:] = struct.pack('f', window.time)
    pipeline.render()
    image.blit()
    ctx.end_frame()
