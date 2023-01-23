import glm
import zengl

from utils import Camera, Window, read_file, read_vertices, set_uniform_glm

window = Window()
camera = Camera()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

vertex_buffer = ctx.buffer(read_vertices('cube.json'))

lighting_shader = ctx.pipeline(
    vertex_shader=read_file('1.colors.vs'),
    fragment_shader=read_file('1.colors.fs'),
    uniforms='all',
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f', 0),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f'),
)

light_cube_shader = ctx.pipeline(
    vertex_shader=read_file('1.light_cube.vs'),
    fragment_shader=read_file('1.light_cube.fs'),
    uniforms='all',
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f', 0),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f'),
)

light_pos = glm.vec3(1.2, 1.0, 2.0)

while window.update():
    camera.update(window)

    set_uniform_glm(lighting_shader, 'projection', camera.projection_matrix)
    set_uniform_glm(lighting_shader, 'view', camera.view_matrix)
    set_uniform_glm(lighting_shader, 'model', glm.mat4(1.0))

    set_uniform_glm(lighting_shader, 'objectColor', glm.vec3(1.0, 0.5, 0.31))
    set_uniform_glm(lighting_shader, 'lightColor', glm.vec3(1.0, 1.0, 1.0))

    set_uniform_glm(light_cube_shader, 'projection', camera.projection_matrix)
    set_uniform_glm(light_cube_shader, 'view', camera.view_matrix)
    set_uniform_glm(light_cube_shader, 'model', glm.scale(glm.translate(glm.mat4(1.0), light_pos), glm.vec3(0.2)))

    image.clear()
    depth.clear()
    lighting_shader.run()
    light_cube_shader.run()
    image.blit()
