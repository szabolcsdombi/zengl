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
    vertex_shader=read_file('6.multiple_lights.vs'),
    fragment_shader=read_file('6.multiple_lights.fs'),
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
    vertex_shader=read_file('6.light_cube.vs'),
    fragment_shader=read_file('6.light_cube.fs'),
    uniforms='all',
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, -1, -1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

point_light_positions = [
    glm.vec3(0.7, 0.2, 2.0),
    glm.vec3(2.3, -3.3, -4.0),
    glm.vec3(-4.0, 2.0, -12.0),
    glm.vec3(0.0, 0.0, -3.0),
]

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

point_lights = [
    glm.vec3(0.7, 0.2, 2.0),
    glm.vec3(2.3, -3.3, -4.0),
    glm.vec3(-4.0, 2.0, -12.0),
    glm.vec3(0.0, 0.0, -3.0),
]

while window.update():
    camera.update(window)

    set_uniform_glm(lighting_shader, 'projection', camera.projection_matrix)
    set_uniform_glm(lighting_shader, 'view', camera.view_matrix)

    set_uniform_float(lighting_shader, 'material.shininess', 32.0)
    set_uniform_glm(lighting_shader, 'viewPos', camera.position)

    set_uniform_glm(lighting_shader, 'dirLight.direction', glm.vec3(-0.2, -1.0, -0.3))
    set_uniform_glm(lighting_shader, 'dirLight.ambient', glm.vec3(0.05, 0.05, 0.05))
    set_uniform_glm(lighting_shader, 'dirLight.diffuse', glm.vec3(0.4, 0.4, 0.4))
    set_uniform_glm(lighting_shader, 'dirLight.specular', glm.vec3(0.5, 0.5, 0.5))

    # point light 1
    set_uniform_glm(lighting_shader, 'pointLights[0].position', point_lights[0])
    set_uniform_glm(lighting_shader, 'pointLights[0].ambient', glm.vec3(0.05, 0.05, 0.05))
    set_uniform_glm(lighting_shader, 'pointLights[0].diffuse', glm.vec3(0.8, 0.8, 0.8))
    set_uniform_glm(lighting_shader, 'pointLights[0].specular', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_float(lighting_shader, 'pointLights[0].constant', 1.0)
    set_uniform_float(lighting_shader, 'pointLights[0].linear', 0.09)
    set_uniform_float(lighting_shader, 'pointLights[0].quadratic', 0.032)

    # point light 2
    set_uniform_glm(lighting_shader, 'pointLights[1].position', point_lights[1])
    set_uniform_glm(lighting_shader, 'pointLights[1].ambient', glm.vec3(0.05, 0.05, 0.05))
    set_uniform_glm(lighting_shader, 'pointLights[1].diffuse', glm.vec3(0.8, 0.8, 0.8))
    set_uniform_glm(lighting_shader, 'pointLights[1].specular', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_float(lighting_shader, 'pointLights[1].constant', 1.0)
    set_uniform_float(lighting_shader, 'pointLights[1].linear', 0.09)
    set_uniform_float(lighting_shader, 'pointLights[1].quadratic', 0.032)

    # point light 3
    set_uniform_glm(lighting_shader, 'pointLights[2].position', point_lights[2])
    set_uniform_glm(lighting_shader, 'pointLights[2].ambient', glm.vec3(0.05, 0.05, 0.05))
    set_uniform_glm(lighting_shader, 'pointLights[2].diffuse', glm.vec3(0.8, 0.8, 0.8))
    set_uniform_glm(lighting_shader, 'pointLights[2].specular', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_float(lighting_shader, 'pointLights[2].constant', 1.0)
    set_uniform_float(lighting_shader, 'pointLights[2].linear', 0.09)
    set_uniform_float(lighting_shader, 'pointLights[2].quadratic', 0.032)

    # point light 4
    set_uniform_glm(lighting_shader, 'pointLights[3].position', point_lights[3])
    set_uniform_glm(lighting_shader, 'pointLights[3].ambient', glm.vec3(0.05, 0.05, 0.05))
    set_uniform_glm(lighting_shader, 'pointLights[3].diffuse', glm.vec3(0.8, 0.8, 0.8))
    set_uniform_glm(lighting_shader, 'pointLights[3].specular', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_float(lighting_shader, 'pointLights[3].constant', 1.0)
    set_uniform_float(lighting_shader, 'pointLights[3].linear', 0.09)
    set_uniform_float(lighting_shader, 'pointLights[3].quadratic', 0.032)

    # spotLight
    set_uniform_glm(lighting_shader, 'spotLight.position', camera.position)
    set_uniform_glm(lighting_shader, 'spotLight.direction', camera.target - camera.position)
    set_uniform_glm(lighting_shader, 'spotLight.ambient', glm.vec3(0.0, 0.0, 0.0))
    set_uniform_glm(lighting_shader, 'spotLight.diffuse', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_glm(lighting_shader, 'spotLight.specular', glm.vec3(1.0, 1.0, 1.0))
    set_uniform_float(lighting_shader, 'spotLight.constant', 1.0)
    set_uniform_float(lighting_shader, 'spotLight.linear', 0.09)
    set_uniform_float(lighting_shader, 'spotLight.quadratic', 0.032)
    set_uniform_float(lighting_shader, 'spotLight.cutOff', glm.cos(glm.radians(12.5)))
    set_uniform_float(lighting_shader, 'spotLight.outerCutOff', glm.cos(glm.radians(15.0)))

    set_uniform_glm(light_cube_shader, 'projection', camera.projection_matrix)
    set_uniform_glm(light_cube_shader, 'view', camera.view_matrix)

    image.clear()
    depth.clear()

    for position, angle in cubes:
        model = glm.mat4(1.0)
        model = glm.translate(model, position)
        model = glm.rotate(model, glm.radians(angle), glm.vec3(1.0, 0.3, 0.5))
        set_uniform_glm(lighting_shader, 'model', model)
        lighting_shader.render()

    for light_pos in point_lights:
        set_uniform_glm(light_cube_shader, 'model', glm.scale(glm.translate(glm.mat4(1.0), light_pos), glm.vec3(0.2)))
        light_cube_shader.render()

    image.blit()
