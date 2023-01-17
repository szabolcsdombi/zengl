import numpy as np
import zengl

from window import Window

window = Window((512, 512))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')
temp = ctx.image(window.size, 'rgba8unorm')

ctx.includes['size'] = 'ivec2 SIZE = ivec2(512);'

scene = ctx.pipeline(
    vertex_shader='''
        #version 450 core

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
        #version 450 core

        layout (binding = 0) uniform sampler2D Texture;

        #include "size"

        layout (location = 0) out vec4 out_color;

        int c(int x, int y) {
            ivec2 at = (ivec2(gl_FragCoord.xy) + ivec2(x, y) + SIZE) % SIZE;
            return texelFetch(Texture, at, 0).r < 0.5 ? 1 : 0;
        }

        void main() {
            float res;
            int neighbours = c(-1, -1) + c(-1, 0) + c(0, 1) + c(0, -1) + c(-1, 1) + c(1, -1) + c(1, 0) + c(1, 1);
            if (c(0, 0) == 1) {
                res = (neighbours == 2 || neighbours == 3) ? 0.0 : 1.0;
            } else {
                res = (neighbours == 3) ? 0.0 : 1.0;
            }
            out_color = vec4(res, res, res, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': temp,
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

cells = np.random.randint(0, 2, 512 * 512)
image.write(np.array([
    cells * 255,
    cells * 255,
    cells * 255,
    cells * 255,
]).T.astype('u1').tobytes())

while window.update():
    image.blit(temp)
    scene.run()
    temp.blit()
