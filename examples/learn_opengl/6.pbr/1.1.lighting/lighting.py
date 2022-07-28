import glm
import zengl

from utils import Camera, Window, download, read_file, set_uniform_float, set_uniform_glm

window = Window()
camera = Camera(1.87, 0.27, 27.0)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

vertex_buffer = ctx.buffer(read_file(download('sphere.mesh'), 'rb'))

pipeline = ctx.pipeline(
    vertex_shader=read_file('1.1.pbr.vs'),
    fragment_shader=read_file('1.1.pbr.fs'),
    uniforms={
        'projection': [0.0] * 16,
        'view': [0.0] * 16,
        'model': [0.0] * 16,
        'lightPositions': [
            (-10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (-10.0, -10.0, 10.0),
            (10.0, -10.0, 10.0),
        ],
        'lightColors': [
            (300.0, 300.0, 300.0),
            (300.0, 300.0, 300.0),
            (300.0, 300.0, 300.0),
            (300.0, 300.0, 300.0),
        ],
        'camPos': (0.0, 0.0, 0.0),
        'albedo': (0.5, 0.0, 0.0),
        'ao': 1.0,
        'metallic': 1.0,
        'roughness': 1.0,
    },
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f 3f 2f', 0, -1, -1, 1, -1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 3f 3f 2f'),
)

num_columns = 7
num_rows = 7
spacing = 2.5

while window.update():
    camera.update(window)

    set_uniform_glm(pipeline, 'projection', camera.projection_matrix)
    set_uniform_glm(pipeline, 'view', camera.view_matrix)
    set_uniform_glm(pipeline, 'camPos', camera.position)

    image.clear()
    depth.clear()

    for row in range(num_rows):
        for col in range(num_columns):
            set_uniform_float(pipeline, 'metallic', row / num_rows)
            set_uniform_float(pipeline, 'roughness', glm.clamp(col / num_columns, 0.05, 1.0))
            model = glm.mat4(1.0)
            model = glm.translate(model, glm.vec3(
                (col - (num_columns // 2)) * spacing,
                (row - (num_rows // 2)) * spacing,
                0.0,
            ))
            set_uniform_glm(pipeline, 'model', model)
            pipeline.render()

    image.blit()
