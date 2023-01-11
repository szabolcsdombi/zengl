import numpy as np
import zengl

from window import Window

window = Window((512, 512))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

N = 16

points_data = (np.array([
    np.repeat(np.linspace(-0.7, 0.7, N), N),
    np.tile(np.linspace(-0.7, 0.7, N), N),
]).T + np.random.uniform(-0.01, 0.01, (N * N, 2))).astype('f4')

edges_data = np.full((N * N, 8), -1, 'i4')

for i in range(N):
    for j in range(N):
        t = []
        if i > 0:
            t.append((i - 1) * N + j)
        if j > 0:
            t.append(i * N + (j - 1))
        if i < N - 1:
            t.append((i + 1) * N + j)
        if j < N - 1:
            t.append(i * N + (j + 1))
        if i > 0 and j > 0:
            t.append((i - 1) * N + (j - 1))
        if i < N - 1 and j < N - 1:
            t.append((i + 1) * N + (j + 1))
        if i > 0 and j < N - 1:
            t.append((i - 1) * N + (j + 1))
        if i < N - 1 and j > 0:
            t.append((i + 1) * N + (j - 1))
        edges_data[i * N + j] = (t + [-1] * 8)[:8]

edge_lengths_data = np.zeros((N * N, 8), 'f4')
for i in range(N):
    for j in range(N):
        for k in range(8):
            a = i * N + j
            b = edges_data[a, k]
            if b < 0:
                edge_lengths_data[a, k] = 0.0
            else:
                edge_lengths_data[a, k] = np.sqrt(np.sum((points_data[a] - points_data[b]) ** 2))

points = ctx.image((N, N), 'rg32float')
points_temp1 = ctx.image((N, N), 'rg32float')
points_temp2 = ctx.image((N, N), 'rg32float')
points.write(points_data)

edges = ctx.image((8, N * N), 'r32sint')
edges.write(edges_data)

edge_lengths = ctx.image((8, N * N), 'r32sint')
edge_lengths.write(edge_lengths_data)

ctx.includes['get_point'] = f'''
    const int N = {N};
    vec2 get_point(int idx) {{
        return texelFetch(Points, ivec2(idx % {N}, idx / {N}), 0).xy;
    }}
'''

edges_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (binding = 0) uniform sampler2D Points;
        layout (binding = 1) uniform isampler2D Edges;

        #include "get_point"

        out vec3 v_color;

        void main() {
            if (gl_VertexID % 2 == 1) {
                int a = texelFetch(Edges, ivec2(gl_VertexID / 2, gl_InstanceID), 0).r;
                if (a < 0) {
                    a = gl_InstanceID;
                }
                gl_Position = vec4(get_point(a), 0.0, 1.0);
            } else {
                gl_Position = vec4(get_point(gl_InstanceID), 0.0, 1.0);
            }
            v_color = vec3(0.0, 0.0, 0.0);
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
    layout=[
        {
            'name': 'Points',
            'binding': 0,
        },
        {
            'name': 'Edges',
            'binding': 1,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': points,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': edges,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image],
    topology='lines',
    vertex_count=16,
    instance_count=N * N,
)

points_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (binding = 0) uniform sampler2D Points;

        #include "get_point"

        out vec3 v_color;

        void main() {
            gl_PointSize = 5.0;
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
            'image': points,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image],
    topology='points',
    vertex_count=N * N,
)

move_points_pipeline = ctx.pipeline(
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

        layout (binding = 0) uniform sampler2D PrevPoints;
        layout (binding = 1) uniform sampler2D Points;

        layout (location = 0) out vec2 out_point;

        void main() {
            vec2 prev_point = texelFetch(PrevPoints, ivec2(gl_FragCoord.xy), 0).xy;
            vec2 point = texelFetch(Points, ivec2(gl_FragCoord.xy), 0).xy;
            vec2 velocity = point - prev_point + vec2(0.0, -0.0001);
            out_point = point + velocity;
            if (out_point.y < -0.9) {
                out_point.y = -0.9;
            }
        }
    ''',
    layout=[
        {
            'name': 'PrevPoints',
            'binding': 0,
        },
        {
            'name': 'Points',
            'binding': 1,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': points_temp1,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': points_temp2,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[points],
    topology='triangles',
    vertex_count=3,
)

constraint_edges_pipeline = ctx.pipeline(
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

        layout (binding = 0) uniform sampler2D Points;
        layout (binding = 1) uniform isampler2D Edges;
        layout (binding = 2) uniform sampler2D EdgeLengths;

        #include "get_point"

        layout (location = 0) out vec2 out_point;

        void main() {
            int point_edges = int(gl_FragCoord.x) + int(gl_FragCoord.y) * N;
            vec2 point = texelFetch(Points, ivec2(gl_FragCoord.xy), 0).xy;
            vec2 new_point = point;
            for (int i = 0; i < 8; ++i) {
                int a = texelFetch(Edges, ivec2(i, point_edges), 0).r;
                if (a < 0) {
                    break;
                }
                vec2 other_point = get_point(a);
                vec2 dir = point - other_point;
                float lng = length(dir);
                float target_lng = texelFetch(EdgeLengths, ivec2(i, point_edges), 0).r;
                new_point += (dir / lng) * (target_lng - lng) * 0.1;
            }
            out_point = new_point;
        }
    ''',
    layout=[
        {
            'name': 'Points',
            'binding': 0,
        },
        {
            'name': 'Edges',
            'binding': 1,
        },
        {
            'name': 'EdgeLengths',
            'binding': 2,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': points_temp1,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': edges,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
        {
            'type': 'sampler',
            'binding': 2,
            'image': edge_lengths,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[points],
    topology='triangles',
    vertex_count=3,
)

points.blit(points_temp1)
points.blit(points_temp2)

while window.update():
    image.clear()
    points_temp2.blit(points_temp1)
    points.blit(points_temp2)
    move_points_pipeline.render()
    points.blit(points_temp1)
    constraint_edges_pipeline.render()
    edges_pipeline.render()
    points_pipeline.render()
    image.blit()
