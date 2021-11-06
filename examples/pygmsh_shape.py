import pygmsh
import numpy as np
import zengl

from window import Window

window = Window(1280, 720)
ctx = zengl.context()
image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

with pygmsh.geo.Geometry() as geom:
    lcar = 0.1
    p1 = geom.add_point([0.0, 0.0], lcar)
    p2 = geom.add_point([1.0, 0.0], lcar)
    p3 = geom.add_point([1.0, 0.5], lcar)
    p4 = geom.add_point([1.0, 1.0], lcar)
    s1 = geom.add_bspline([p1, p2, p3, p4])

    p2 = geom.add_point([0.0, 1.0], lcar)
    p3 = geom.add_point([0.5, 1.0], lcar)
    s2 = geom.add_spline([p4, p3, p2, p1])

    ll = geom.add_curve_loop([s1, s2])
    pl = geom.add_plane_surface(ll)

    mesh = geom.generate_mesh()

vertex_data = mesh.points[:, :2]
index_data = mesh.cells[1].data

vertex_buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
index_buffer = ctx.buffer(index_data.astype('i4').tobytes())

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (location = 0) in vec2 in_vert;

        out vec3 v_color;

        void main() {
            gl_Position = vec4(in_vert - 0.5, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '2f', 0),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
)


@window.render
def render():
    image.clear()
    triangle.render()
    image.blit()


window.run()
