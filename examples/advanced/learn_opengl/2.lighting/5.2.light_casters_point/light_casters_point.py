import glm
import zengl
from PIL import Image

from utils import Camera, Window, download, read_file, read_vertices, set_uniform_float, set_uniform_glm

window = Window()
camera = Camera()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

vertex_buffer = ctx.buffer(read_vertices('cube.json'))

img = Image.open(download('container2.png')).convert('RGBA')
diffuse_map = ctx.image(img.size, 'rgba8unorm', img.tobytes())

img = Image.open(download('container2_specular.png')).convert('RGBA')
specular_map = ctx.image(img.size, 'rgba8unorm', img.tobytes())

lighting_shader = ctx.pipeline(
    vertex_shader=read_file('5.2.light_casters.vs'),
    fragment_shader=read_file('5.2.light_casters.fs'),
    layout=[
        {
            'name': 'material.diffuse',
            'binding': 0,
        },
        {
            'name': 'material.specular',
            'binding': 1,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': diffuse_map,
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': specular_map,
        },
    ],
    uniforms='all',
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

light_cube_shader = ctx.pipeline(
    vertex_shader=read_file('5.2.light_cube.vs'),
    fragment_shader=read_file('5.2.light_cube.fs'),
    uniforms='all',
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, -1, -1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

light_pos = glm.vec3(1.0, 1.0, 1.0)
light_color = glm.vec3(1.0, 1.0, 1.0)

cubes = [
    [glm.vec3(0.0, 0.0, 0.0), 0.0],
    [glm.vec3(2.0, 5.0, -15.0), 20.0],
    [glm.vec3(-1.5, -2.2, -2.5), 40.0],
    [glm.vec3(-3.8, -2.0, -12.3), 60.0],
    [glm.vec3(2.4, -0.4, -3.5), 80.0],
    [glm.vec3(-1.7, 3.0, -7.5), 100.0],
    [glm.vec3(1.3, -2.0, -2.5), 120.0],
    [glm.vec3(1.5, 2.0, -2.5), 140.0],
    [glm.vec3(1.5, 0.2, -1.5), 160.0],
    [glm.vec3(-1.3, 1.0, -1.5), 180.0],
]

while window.update():
    camera.update(window)

    set_uniform_glm(lighting_shader, 'projection', camera.projection_matrix)
    set_uniform_glm(lighting_shader, 'view', camera.view_matrix)

    set_uniform_float(lighting_shader, 'material.shininess', 32.0)
    set_uniform_glm(lighting_shader, 'viewPos', camera.position)

    set_uniform_glm(lighting_shader, 'light.position', light_pos)
    set_uniform_glm(lighting_shader, 'light.ambient', glm.vec3(0.2, 0.2, 0.2))
    set_uniform_glm(lighting_shader, 'light.diffuse', glm.vec3(0.5, 0.5, 0.5))
    set_uniform_glm(lighting_shader, 'light.specular', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_float(lighting_shader, 'light.constant', 1.0)
    set_uniform_float(lighting_shader, 'light.linear', 0.09)
    set_uniform_float(lighting_shader, 'light.quadratic', 0.032)

    set_uniform_glm(light_cube_shader, 'projection', camera.projection_matrix)
    set_uniform_glm(light_cube_shader, 'view', camera.view_matrix)
    set_uniform_glm(light_cube_shader, 'model', glm.scale(glm.translate(glm.mat4(1.0), light_pos), glm.vec3(0.2)))

    image.clear()
    depth.clear()

    for position, angle in cubes:
        model = glm.mat4(1.0)
        model = glm.translate(model, position)
        model = glm.rotate(model, glm.radians(angle), glm.vec3(1.0, 0.3, 0.5))
        set_uniform_glm(lighting_shader, 'model', model)
        lighting_shader.render()

    light_cube_shader.render()
    image.blit()
