import glwindow
import zengl


class HelloTriangle:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', samples=samples)
        self.output = self.image if self.image.samples == 1 else self.ctx.image(size, 'rgba8unorm')

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

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
                #version 300 es
                precision highp float;

                in vec3 v_color;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(v_color, 1.0);
                    out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
                }
            ''',
            framebuffer=[self.image],
            topology='triangles',
            vertex_count=3,
        )

    def render(self):
        self.image.clear()
        self.pipeline.render()
        if self.image != self.output:
            self.image.blit(self.output)


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene = HelloTriangle(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
