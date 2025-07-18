import sys

import numpy as np
import pygame
import zengl
import zengl_extras

N = 128

zengl_extras.init()

pygame.init()
pygame.display.set_mode((512, 512), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
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
        #version 300 es
        precision highp float;

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
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            vec2 cursor;
        };

        uniform sampler2D PreviousPoints;
        uniform sampler2D Points;

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
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'PreviousPoints',
            'binding': 0,
        },
        {
            'name': 'Points',
            'binding': 1,
        },
    ],
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
        #version 300 es
        precision highp float;

        uniform sampler2D Points;

        #include "get_point"

        out vec3 v_color;

        void main() {
            gl_PointSize = 2.0;
            gl_Position = vec4(get_point(gl_VertexID), 0.0, 1.0);
            v_color = vec3(0.0, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Points',
            'binding': 0,
        },
    ],
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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    mouse_pos = pygame.mouse.get_pos()
    cx = cx * 0.95 + (mouse_pos[0] / window_size[0] * 2.0 - 1.0) * 0.05
    cy = cy * 0.95 + -(mouse_pos[1] / window_size[1] * 2.0 - 1.0) * 0.05
    ctx.new_frame()
    uniform_buffer.write(np.array([cx, cy, 0.0, 0.0], 'f4'))
    image.clear()
    simulate_pipeline.render()
    render_pipeline.render()
    points[1].blit(points[0])
    points[2].blit(points[1])
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
