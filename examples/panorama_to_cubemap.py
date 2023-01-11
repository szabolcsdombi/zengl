import zengl
from PIL import Image

import assets

ctx = zengl.context(zengl.loader(headless=True))

img = Image.open(assets.get('comfy_cafe.jpg'))  # https://polyhaven.com/a/comfy_cafe
texture = ctx.image(img.size, 'rgba8unorm', img.convert('RGBA').tobytes())

image = ctx.image((1024, 1024), 'rgba8unorm')


def face_pipeline(face):
    uv_to_dir = [
        'vec3(1.0, uv.x, uv.y)',
        'vec3(-1.0, -uv.x, uv.y)',
        'vec3(uv.x, -uv.y, -1.0)',
        'vec3(uv.x, uv.y, 1.0)',
        'vec3(uv.x, -1.0, uv.y)',
        'vec3(-uv.x, 1.0, uv.y)',
    ]

    ctx.includes['uv_to_dir'] = f'''
        vec3 uv_to_dir(vec2 uv) {{
            return {uv_to_dir[face]};
        }}
    '''

    return ctx.pipeline(
        vertex_shader='''
            #version 450 core

            vec2 vertices[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(3.0, -1.0),
                vec2(-1.0, 3.0)
            );

            out vec2 v_texcoord;

            void main() {
                gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
                v_texcoord = vertices[gl_VertexID];
            }
        ''',
        fragment_shader='''
            #version 450 core

            #include "uv_to_dir"

            const float pi = 3.14159265358979323;
            layout (binding = 0) uniform sampler2D Texture;

            in vec2 v_texcoord;

            layout (location = 0) out vec4 out_color;

            vec2 dir_to_uv(vec3 dir) {
                return vec2(0.5 + 0.5 * atan(dir.z, dir.x) / pi, 1.0 - acos(dir.y) / pi);
            }

            void main() {
                vec3 scan = uv_to_dir(v_texcoord);
                vec3 direction = normalize(scan);
                vec2 uv = dir_to_uv(direction);
                out_color = texture(Texture, uv);
            }
        ''',
        layout=[
            {
                'name': 'Texture',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'sampler',
                'binding': 0,
                'image': texture,
            },
        ],
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )


pipelines = [(i, face_pipeline(i)) for i in range(6)]

for face, pipeline in pipelines:
    pipeline.render()
    img = Image.frombuffer('RGBA', image.size, image.read(), 'raw', 'RGBA', 0, -1)
    img.save(f'downloads/skybox_{face}.png')
