import numpy as np
import zengl

from window import Window

window = Window(1280, 720)
ctx = zengl.context(zengl.loader())

ctx.includes['parameters'] = '''
    vec2 Size = vec2(1280.0, 720.0);
    vec2 Center = vec2(0.49, 0.32);
    int Iter = 100;
'''

image = ctx.image(window.size, 'rgba8unorm', samples=4)

scene = ctx.pipeline(
    vertex_shader='''
        #version 330

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        #include "parameters"

        layout (location = 0) out vec4 out_color;

        void main() {
            vec2 z = vec2(5.0, 3.0) * (gl_FragCoord.xy / Size - 0.5);
            vec2 c = Center;
            int i;
            for(i = 0; i < Iter; i++) {
                vec2 v = vec2(
                    (z.x * z.x - z.y * z.y) + c.x,
                    (z.y * z.x + z.x * z.y) + c.y
                );
                if (dot(v, v) > 4.0) break;
                z = v;
            }
            float cm = fract((i == Iter ? 0.0 : float(i)) * 10 / Iter);
            out_color = vec4(
                fract(cm + 0.0 / 3.0),
                fract(cm + 1.0 / 3.0),
                fract(cm + 2.0 / 3.0),
                1.0
            );
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)


@window.render
def render():
    image.clear()
    scene.render()
    image.blit()


window.run()
