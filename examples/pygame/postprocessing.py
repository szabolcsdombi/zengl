import random
import struct

import pygame
import zengl

pygame.init()
pygame.display.set_mode((640, 480), flags=pygame.OPENGL | pygame.DOUBLEBUF)
screen = pygame.surface.Surface((320, 240))

ctx = zengl.context()
image = ctx.image((320, 240), "rgba8unorm")
pipeline = ctx.pipeline(
    vertex_shader="""
        #version 300 es
        precision highp float;

        vec2 vertices[4] = vec2[](
            vec2(-1.0, -1.0),
            vec2(-1.0, 1.0),
            vec2(1.0, -1.0),
            vec2(1.0, 1.0)
        );

        out vec2 vertex;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            vertex = vertices[gl_VertexID];
        }
    """,
    fragment_shader="""
        #version 300 es
        precision highp float;

        uniform float time;
        uniform vec2 screen_size;
        uniform sampler2D Texture;

        in vec2 vertex;
        out vec4 out_color;

        float hash13(vec3 p3) {
            p3 = fract(p3 * 0.1031);
            p3 += dot(p3, p3.zyx + 31.32);
            return fract((p3.x + p3.y) * p3.z);
        }

        vec2 hash23(vec3 p3) {
            p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.xx + p3.yz) * p3.zy);
        }

        vec3 sample_screen(vec2 uv) {
            return vec3(
                texture(Texture, uv - vec2(0.002, 0.0)).r,
                texture(Texture, uv).g,
                texture(Texture, uv + vec2(0.002, 0.0)).b
            );
        }

        vec3 screen(vec2 vertex) {
            if (abs(vertex.x) > 1.001 || abs(vertex.y) > 1.001) {
                return vec3(0.0);
            }
            vec2 uv = vertex * 0.5 + 0.5;
            vec3 color = sample_screen(uv);
            float noise = hash13(floor(vec3(uv * screen_size, floor(time))));
            float scanline = sin(uv.y * screen_size.y * 3.141592 + time * 0.05) * 0.1 + 0.9;
            color *= 1.0 + (noise - 0.5) * 0.1;
            color = color * scanline * 1.05 + 0.05;
            return color;
        }

        void main() {
            vec2 v = vertex * (1.0 + pow(abs(vertex.yx), vec2(2.0)) * 0.1);
            vec3 color = vec3(0.0);
            for (int i = 0; i < 16; ++i) {
                vec2 offset = hash23(vec3(v * screen_size, float(i))) - 0.5;
                color += screen(v * 1.01 + offset / screen_size);
            }
            color /= 16.0;
            out_color = vec4(color, 1.0);
        }
    """,
    layout=[
        {
            "name": "Texture",
            "binding": 0,
        },
    ],
    resources=[
        {
            "type": "sampler",
            "binding": 0,
            "image": image,
            "min_filter": "nearest",
            "mag_filter": "nearest",
            "wrap_x": "clamp_to_edge",
            "wrap_y": "clamp_to_edge",
        },
    ],
    uniforms={
        "time": 0.0,
        "screen_size": (320.0, 240.0),
    },
    framebuffer=None,
    viewport=(0, 0, 640, 480),
    topology="triangle_strip",
    vertex_count=4,
)


def make_square(size, color):
    surface = pygame.surface.Surface((size, size), pygame.SRCALPHA)
    surface.fill(color)
    return surface


squares = []
for _ in range(20):
    size = random.randint(20, 80)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    square = make_square(size, color)
    squares.append({
        "position": (random.randint(0, 320), random.randint(0, 240)),
        "rotation": random.randint(0, 360),
        "surface": square,
    })


clock = pygame.Clock()

space_down = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key == pygame.K_SPACE:
                space_down = event.type == pygame.KEYDOWN

    screen.fill((0, 0, 0))

    for square in squares:
        square["rotation"] += 1
        if square["rotation"] > 360:
            square["rotation"] = 0
        rotated_surface = pygame.transform.rotate(square["surface"], square["rotation"])
        width, height = rotated_surface.get_size()
        position = (square["position"][0] - width // 2, square["position"][1] - height // 2)
        screen.blit(rotated_surface, position)

    ctx.new_frame()
    image.write(pygame.image.tobytes(screen, "RGBA", flipped=True))

    if not space_down:
        pipeline.uniforms["time"][:] = struct.pack("f", pygame.time.get_ticks())
        pipeline.render()

    else:
        image.blit(size=(640, 480))

    ctx.end_frame()

    pygame.display.flip()
    clock.tick(60)
