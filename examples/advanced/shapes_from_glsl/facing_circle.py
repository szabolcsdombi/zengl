import zengl

from defaults import defaults
from grid import grid_pipeline
from window import Window

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

ctx.includes['defaults'] = defaults

grid = grid_pipeline(ctx, [image, depth])

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "defaults"

        vec2 vertices[16] = vec2[](
            vec2(1.0000, 0.0000),
            vec2(0.9239, 0.3827),
            vec2(0.7071, 0.7071),
            vec2(0.3827, 0.9239),
            vec2(0.0000, 1.0000),
            vec2(-0.3827, 0.9239),
            vec2(-0.7071, 0.7071),
            vec2(-0.9239, 0.3827),
            vec2(-1.0000, 0.0000),
            vec2(-0.9239, -0.3827),
            vec2(-0.7071, -0.7071),
            vec2(-0.3827, -0.9239),
            vec2(0.0000, -1.0000),
            vec2(0.3827, -0.9239),
            vec2(0.7071, -0.7071),
            vec2(0.9239, -0.3827)
        );

        vec3 position = vec3(0.0, 0.0, 0.0);
        vec3 up = vec3(0.0, 0.0, 1.0);

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            v_normal = normalize(eye.xyz - position);
            vec3 tangent = normalize(cross(up, v_normal));
            vec3 bitangent = cross(v_normal, tangent);
            v_vertex = position + tangent * vertices[gl_VertexID].x + bitangent * vertices[gl_VertexID].y;
            v_texcoord = vertices[gl_VertexID] + 0.5;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        #include "defaults"

        in vec3 v_normal;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = dot(normalize(light.xyz), normalize(v_normal)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    framebuffer=[image, depth],
    topology='triangle_fan',
    cull_face='back',
    vertex_count=16,
)

while window.update():
    image.clear()
    depth.clear()
    grid.render()
    pipeline.render()
    image.blit()
