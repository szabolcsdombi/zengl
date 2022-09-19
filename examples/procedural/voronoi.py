import zengl

from window import Window

window = Window((720, 720))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')

voronoi = ctx.pipeline(
    vertex_shader='''
        #version 330
        vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        layout (location = 0) out vec4 out_color;

        uniform vec2 resolution;

        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        vec3 hash32(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yxz + 33.33);
            return fract((p3.xxy + p3.yzz) * p3.zyx);
        }

        vec3 voronoi(vec2 uv) {
            vec3 res;
            float dist = 100.0;
            for (int i = 0; i < 9; ++i) {
                vec2 offset = vec2(float(i % 3), float(i / 3)) - 1.0;
                vec3 cell = hash32(floor(uv) + offset);
                cell.xy += floor(uv) + offset;
                float lng = length(uv - cell.xy);
                if (dist > lng) {
                    dist = lng;
                    res = cell;
                }
            }
            return res;
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution;
            uv = uv * 10;

            vec3 cell = voronoi(uv);
            if (length(uv - cell.xy) < 0.05) {
                out_color = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }

            vec3 color = hsv2rgb(vec3(cell.z, 0.5, 1.0));
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
    voronoi.render()
    image.blit()
