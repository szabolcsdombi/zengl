import glwindow
import zengl
from monkey import Monkey


class FXAA:
    def __init__(self, src: zengl.Image):
        self.ctx = zengl.context()

        width, height = src.size
        self.src = src
        self.temp = self.ctx.image((width // 2, height // 2), 'rgba8unorm')
        self.output = self.ctx.image(src.size, 'rgba8unorm')

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                vec2 positions[3] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2(3.0, -1.0),
                    vec2(-1.0, 3.0)
                );

                out vec2 v_textcoord;

                void main() {
                    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
                    v_textcoord = positions[gl_VertexID] * 0.5 + 0.5;
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                uniform sampler2D Texture1;
                uniform sampler2D Texture2;

                in vec2 v_textcoord;

                layout (location = 0) out vec4 out_color;

                float luminance(vec3 color) {
                    return dot(color, vec3(0.2126, 0.7152, 0.0722));
                }

                void main() {
                    vec3 color = texelFetch(Texture1, ivec2(gl_FragCoord.xy), 0).rgb;
                    vec3 blur = texture(Texture2, v_textcoord).rgb;
                    float diff = abs(luminance(color) - luminance(blur));
                    out_color = vec4(mix(color, blur, diff), 1.0);
                    // out_color.rbg *= 0.1;
                    // out_color.r += diff;
                }
            ''',
            layout=[
                {
                    'name': 'Texture1',
                    'binding': 0,
                },
                {
                    'name': 'Texture2',
                    'binding': 1,
                },
            ],
            resources=[
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.src,
                },
                {
                    'type': 'sampler',
                    'binding': 1,
                    'image': self.temp,
                    'min_filter': 'linear',
                    'mag_filter': 'linear',
                },
            ],
            framebuffer=[self.output],
            topology='triangles',
            vertex_count=3,
        )

    def render(self):
        self.src.blit(self.temp)
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())

        self.scene = Monkey(self.wnd.size, samples=1)
        self.fxaa = FXAA(self.scene.image)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.fxaa.render()
        self.fxaa.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
