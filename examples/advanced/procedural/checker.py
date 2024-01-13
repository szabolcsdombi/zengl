import zengl

from window import Window

window = Window((720, 720))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')

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

        layout (location = 0) out vec4 out_color;

        uniform vec2 resolution;

        struct CheckerTexture {
            float scale;
            vec3 color1;
            vec3 color2;
        };

        float checker_factor(vec2 uv, CheckerTexture mt) {
            vec2 s = sign(fract(uv * mt.scale * 0.5) - 0.5);
            return 0.5 - s.x * s.y * 0.5;
        }

        vec3 checker_color(vec2 uv, CheckerTexture mt) {
            return mix(mt.color1, mt.color2, checker_factor(uv, mt));
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution;

            CheckerTexture mt;
            mt.scale = 5.0;
            mt.color1 = vec3(0.2, 0.2, 0.2);
            mt.color2 = vec3(0.8, 0.8, 0.8);

            vec3 color = checker_color(uv, mt);
            out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    uniforms={
        'resolution': window.size,
    },
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while window.update():
    image.clear()
    pipeline.render()
    image.blit()
