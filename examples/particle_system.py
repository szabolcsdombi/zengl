import numpy as np
import zengl

from window import Window

N = 128

window = Window((512, 512))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

uniform_buffer = ctx.buffer(size=16)

angle = np.random.uniform(0.0, np.pi * 2.0, (N, N))
radius = np.random.uniform(0.2, 0.7, (N, N))
speed = np.random.uniform(0.03, 0.05, (N, N))
points_data = np.array([np.cos(angle) * radius, np.sin(angle) * radius]).transpose(1, 2, 0).astype('f4').copy()
velocity_data = np.array([-np.sin(angle) * speed, np.cos(angle) * speed]).transpose(1, 2, 0).astype('f4').copy()

points = [ctx.image((N, N), 'rg32float') for _ in range(3)]

points[0].write(points_data)
points[1].write(points_data + velocity_data)

ctx.includes['get_point'] = f'''
    const int N = {N};
    vec2 get_point(int idx) {{
        return texelFetch(Points, ivec2(idx % {N}, idx / {N}), 0).xy;
    }}
'''

simulate_pipeline = ctx.pipeline(
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

        layout (std140, binding = 0) uniform Common {
            vec2 cursor;
        };

        layout (binding = 0) uniform sampler2D PreviousPoints;
        layout (binding = 1) uniform sampler2D Points;

        layout (location = 0) out vec2 out_point;

        void main() {
            vec2 prev_point = texelFetch(PreviousPoints, ivec2(gl_FragCoord.xy), 0).xy;
            vec2 point = texelFetch(Points, ivec2(gl_FragCoord.xy), 0).xy;
            vec2 velocity = point - prev_point; // + vec2(0.0, -0.0001);
            vec2 dir = cursor - point;
            float squared_distance = min(dot(dir, dir), 0.1);
            velocity += dir / squared_distance * 0.001;
            float squared_speed = dot(velocity, velocity);
            float max_speed = 0.01;
            if (squared_speed > max_speed) {
                velocity = velocity * (max_speed / sqrt(squared_speed));
            }
            out_point = point + velocity;
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'sampler',
            'binding': 0,
            'image': points[0],
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': points[1],
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[points[2]],
    topology='triangles',
    vertex_count=3,
)

render_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (binding = 0) uniform sampler2D Points;

        #include "get_point"

        out vec3 v_color;

        void main() {
            gl_PointSize = 2.0;
            gl_Position = vec4(get_point(gl_VertexID), 0.0, 1.0);
            v_color = vec3(0.0, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': points[2],
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image],
    topology='points',
    vertex_count=N * N,
)

cx, cy = 0.0, 0.0

while window.update():
    cx = cx * 0.95 + (window.mouse[0] / window.size[0] * 2.0 - 1.0) * 0.05
    cy = cy * 0.95 + (window.mouse[1] / window.size[1] * 2.0 - 1.0) * 0.05
    uniform_buffer.write(np.array([cx, cy, 0.0, 0.0], 'f4'))
    image.clear()
    simulate_pipeline.run()
    render_pipeline.run()
    points[1].blit(points[0])
    points[2].blit(points[1])
    image.blit()
