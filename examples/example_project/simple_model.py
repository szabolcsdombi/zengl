from typing import List

import zengl
from objloader import Obj

from context import Context


class SimpleModel:
    def __init__(self, filename: str):
        self.ctx = Context.context
        model = Obj.open(filename).pack('vx vy vz nx ny nz')
        self.vertex_buffer = self.ctx.buffer(model)

    def pipeline(self, object_state_resource, framebuffer: List[zengl.Image]):
        return self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                #include "main_uniform_buffer"

                layout (std140) uniform ObjectState {
                    vec3 position;
                };

                layout (location = 0) in vec3 in_vert;
                layout (location = 1) in vec3 in_norm;

                out vec3 v_norm;

                void main() {
                    gl_Position = mvp * vec4(position + in_vert, 1.0);
                    v_norm = in_norm;
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec3 v_norm;

                layout (location = 0) out vec4 out_color;

                void main() {
                    vec3 light = vec3(4.0, 3.0, 10.0);
                    float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
                    out_color = vec4(lum, lum, lum, 1.0);
                }
            ''',
            layout=[
                {
                    'name': 'MainUniformBuffer',
                    'binding': 0,
                },
                {
                    'name': 'ObjectState',
                    'binding': 1,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': Context.main_uniform_buffer,
                },
                object_state_resource,
            ],
            framebuffer=framebuffer,
            topology='triangles',
            cull_face='back',
            vertex_buffers=zengl.bind(self.vertex_buffer, '3f 3f', 0, 1),
            vertex_count=self.vertex_buffer.size // zengl.calcsize('3f 3f'),
        )
