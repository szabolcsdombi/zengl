import zengl

from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')
image.clear_value = (1.0, 1.0, 1.0, 1.0)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        out vec3 v_color;

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec3 v_color;

        layout(rgba8, binding = 0) writeonly uniform image2D output_image;

        void main() {
            imageStore(output_image, ivec2(gl_FragCoord.xy), vec4(v_color, 1.0));
        }
    ''',
    resources=[
        {
            'type': 'image',
            'binding': 0,
            'image': image,
        },
    ],
    framebuffer_size=(1280, 720),
    topology='triangles',
    vertex_count=3,
)

image.clear()
pipeline.run()
ctx.barrier()

while window.update():
    image.blit()
