import zengl

from window import Window

window = Window((720, 720))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core
        vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (location = 0) out vec4 out_color;

        uniform vec2 resolution;

        struct BrickTexture {
            float scale;
            vec2 brick_size;
            float offset;
            int offset_frequency;
            float squash;
            int squash_frequency;
            float mortar_size;
            float mortar_smooth;
            float bias;
            vec3 color1;
            vec3 color2;
            vec3 color3;
        };

        float hash12(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        vec2 brick_coordinates(vec2 uv, BrickTexture mt) {
            vec2 b = uv * mt.scale / mt.brick_size;
            if (int(floor(b.y) + 1) % mt.offset_frequency == 0) {
                b.x += mt.offset;
            }
            if (int(floor(b.y) + 1) % mt.squash_frequency == 0) {
                b.x *= mt.squash;
            }
            return b;
        }

        float brick_factor(vec2 uv, BrickTexture mt) {
            vec2 b = brick_coordinates(uv, mt);
            vec2 a = (0.5 - abs(fract(b) - 0.5)) * mt.brick_size;
            return 1.0 - smoothstep(mt.mortar_size * (1.0 - mt.mortar_smooth), mt.mortar_size, min(a.x, a.y));
        }

        vec3 brick_color(vec2 uv, BrickTexture mt) {
            vec2 b = brick_coordinates(uv, mt);
            float f1 = hash12(floor(b));
            float f2 = brick_factor(uv, mt);
            vec3 color = mix(mt.color1, mt.color2, f1);
            return mix(color, mt.color3, f2);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution;

            BrickTexture mt;
            mt.offset = 0.5;
            mt.offset_frequency = 2;
            mt.squash = 1.0;
            mt.squash_frequency = 2;
            mt.scale = 5.0;
            mt.mortar_size = 0.02;
            mt.mortar_smooth = 0.1;
            mt.bias = 0.0;
            mt.brick_size = vec2(0.5, 0.25);
            mt.color1 = vec3(0.2, 0.2, 0.2);
            mt.color2 = vec3(0.8, 0.8, 0.8);
            mt.color3 = vec3(0.0, 0.0, 0.0);

            vec3 color = brick_color(uv, mt);
            out_color = vec4(color, 1.0);
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
