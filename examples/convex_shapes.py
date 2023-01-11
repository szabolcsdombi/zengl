import numpy as np
import zengl
from scipy.spatial import ConvexHull

from window import Window


def make_hull(points):
    hull = ConvexHull(points)
    vertices = hull.points[hull.simplices]
    normals = np.cross(vertices[:, 0] - vertices[:, 1], vertices[:, 0] - vertices[:, 2])
    sign = np.sign(np.sum(vertices[:, 0] * normals, axis=1))
    vertices[np.where(sign < 0.0)] = vertices[np.where(sign < 0.0), ::-1]
    normals = (normals.T / np.sqrt(np.sum(normals * normals, axis=1)) * sign).T
    mesh = np.zeros((hull.simplices.shape[0] * 3, 9), 'f4')
    mesh[:, 0:3] = vertices.reshape(-1, 3)
    mesh[:, 3:6] = np.repeat(normals, 3, axis=0)
    return mesh


def gen_box(width, length, height):
    points = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])
    points *= (width, length, height)
    return points


def gen_cylinder(radius, height, res=32):
    x = np.cos(np.linspace(0.0, np.pi * 4.0, res * 2, endpoint=False))
    y = np.sin(np.linspace(0.0, np.pi * 4.0, res * 2, endpoint=False))
    z = np.repeat([-0.5, 0.5], res)
    return np.array([x, y, z]).T * (radius, radius, height)


def gen_cone(radius, height, res=32):
    x = np.cos(np.linspace(0.0, np.pi * 2.0, res + 1))
    y = np.sin(np.linspace(0.0, np.pi * 2.0, res + 1))
    z = np.full(res + 1, -0.5)
    points = np.array([x, y, z]).T
    points[-1] = [0.0, 0.0, 0.5]
    return points * (radius, radius, height)


def gen_capsule(radius, height, res=16):
    sphere = gen_uvsphere(radius, res)
    offset = [0.0, 0.0, height * 0.5]
    return np.concatenate([sphere - offset, sphere + offset])


def gen_sphere(radius, res=100):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (np.arange(res) / (res - 1.0)) * 2.0
    x = np.cos(phi * np.arange(res)) * np.sqrt(1.0 - y * y)
    z = np.sin(phi * np.arange(res)) * np.sqrt(1.0 - y * y)
    return np.array([x, y, z]).T * radius


def gen_uvsphere(radius, res=16):
    h = np.repeat(np.linspace(0.0, np.pi * 2.0, res * 2, endpoint=False), res - 1)
    v = np.tile(np.linspace(0.0, np.pi, res + 1)[1:-1], res * 2)
    ends = [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
    points = np.concatenate([np.array([np.cos(h) * np.sin(v), np.sin(h) * np.sin(v), np.cos(v)]).T, ends])
    return points * radius


def gen_multisphere(points, res=100):
    return np.concatenate([gen_sphere(r, res) + (x, y, z) for x, y, z, r in points])


def gen_minkowski(a, b):
    return np.tile(a.flatten(), b.shape[0]).reshape(-1, 3) + np.tile(b, a.shape[0]).reshape(-1, 3)


def transform(frame, mesh):
    mesh = mesh.copy()
    for i in range(mesh.shape[0]):
        x, y, z, nx, ny, nz = mesh[i, 0:6]
        mesh[i, 0:3] = (frame @ [x, y, z, 1.0])[:3]
        mesh[i, 3:6] = (frame @ [nx, ny, nz, 0.0])[:3]
    mesh[:, 3:6] = (mesh[:, 3:6].T / np.sqrt(np.sum(mesh[:, 3:6] * mesh[:, 3:6], axis=1))).T
    return mesh


window = Window()
ctx = zengl.context()

# vertex_buffer = ctx.buffer(make_hull(np.random.uniform(-0.5, 0.5, (100, 3))))
# vertex_buffer = ctx.buffer(make_hull(gen_box(1.0, 1.0, 1.0)))
# vertex_buffer = ctx.buffer(make_hull(gen_sphere(1.0)))
# vertex_buffer = ctx.buffer(make_hull(gen_uvsphere(1.0)))
# vertex_buffer = ctx.buffer(make_hull(gen_minkowski(gen_box(0.9, 0.9, 0.9), gen_uvsphere(0.1))))
# vertex_buffer = ctx.buffer(make_hull(gen_cylinder(1.0, 1.0)))
# vertex_buffer = ctx.buffer(make_hull(gen_minkowski(gen_cylinder(1.0, 1.0), gen_uvsphere(0.1))))
# vertex_buffer = ctx.buffer(make_hull(gen_cone(0.5, 1.0)))
# vertex_buffer = ctx.buffer(make_hull(gen_capsule(0.3, 1.0)))
vertex_buffer = ctx.buffer(make_hull(gen_multisphere([[0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 1.0, 0.1]])))

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

uniform_buffer = ctx.buffer(size=80)

shape = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f', 0, 1, -1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 3f'),
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    image.clear()
    depth.clear()
    shape.render()
    image.blit()
