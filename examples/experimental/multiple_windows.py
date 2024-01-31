# pygame-ce == 2.4.0
# Multiple windows using pygame-ce Window(opengl=True) instances
# Steps:
# - ZenGL creates a headless context, this is the main OpenGL context
# - pygame-ce creates multiple windows
#     at the moment the Window(opengl=True) does not create a context
#     this is corrected by calling into sdl2 directly from ctypes
#     please also note that wglMakeCurrent and SDL_GL_MakeCurrent are not interchangeable
# - wglShareLists is used to share OpenGL resources between the window and ZenGL
# - for each window a framebuffer is created (framebuffer objects are not shared)
# - ZenGL renders into different images
# - every window blits its own image to the screen
# - DwmFlush is called for proper vsync control, every window has swap control off

import ctypes
import math
import os
import struct

import pygame
import zengl

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"


def generate_surface(size, text):
    pygame.font.init()
    font = pygame.font.SysFont('Consolas', 64)
    text = font.render(text, False, (0, 0, 255))
    tx, ty = text.get_size()
    surf = pygame.surface.Surface(size)
    surf.fill((255, 255, 255))
    surf.blit(text, ((size[0] - tx) // 2, (size[1] - ty + 10) // 2))
    surf = pygame.transform.flip(surf, False, True)
    return bytes(surf.get_buffer())


def generate_texture(size, text):
    ctx = zengl.context()
    return ctx.image(size, 'rgba8unorm', generate_surface(size, text))


def glfunction(function, fmt, restype=None):
    argtypes = tuple({'i': ctypes.c_int, 'p': ctypes.c_void_p, 'f': ctypes.c_float}[c] for c in fmt)
    return ctypes.WINFUNCTYPE(restype, *argtypes)(function)


class DetectContext:
    def __init__(self):
        ctypes.windll.opengl32.wglGetCurrentDC.restype = ctypes.c_void_p
        ctypes.windll.opengl32.wglGetCurrentContext.restype = ctypes.c_void_p
        self.hdc = ctypes.windll.opengl32.wglGetCurrentDC()
        self.hglrc = ctypes.windll.opengl32.wglGetCurrentContext()

    def make_current(self):
        ctypes.windll.opengl32.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ctypes.windll.opengl32.wglMakeCurrent(self.hdc, self.hglrc)

    def inherit(self, other):
        ctypes.windll.opengl32.wglShareLists.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ctypes.windll.opengl32.wglShareLists(other.hglrc, self.hglrc)


class DetectGL:
    def __init__(self):
        ctypes.windll.opengl32.wglGetProcAddress.argtypes = [ctypes.c_void_p]
        ctypes.windll.opengl32.wglGetProcAddress.restype = ctypes.c_void_p

        glFlush = ctypes.windll.opengl32.glFlush
        glReadBuffer = ctypes.windll.opengl32.glReadBuffer
        glDrawBuffers = ctypes.windll.opengl32.wglGetProcAddress(b'glDrawBuffers')
        glBindFramebuffer = ctypes.windll.opengl32.wglGetProcAddress(b'glBindFramebuffer')
        glGenFramebuffers = ctypes.windll.opengl32.wglGetProcAddress(b'glGenFramebuffers')
        glFramebufferTexture2D = ctypes.windll.opengl32.wglGetProcAddress(b'glFramebufferTexture2D')
        glFramebufferRenderbuffer = ctypes.windll.opengl32.wglGetProcAddress(b'glFramebufferRenderbuffer')
        glBlitFramebuffer = ctypes.windll.opengl32.wglGetProcAddress(b'glBlitFramebuffer')
        wglSwapIntervalEXT = ctypes.windll.opengl32.wglGetProcAddress(b'wglSwapIntervalEXT')

        self.glFlush = glfunction(glFlush, '')
        self.glReadBuffer = glfunction(glReadBuffer, 'i')
        self.glDrawBuffers = glfunction(glDrawBuffers, 'ip')
        self.glBindFramebuffer = glfunction(glBindFramebuffer, 'ii')
        self.glGenFramebuffers = glfunction(glGenFramebuffers, 'ip')
        self.glFramebufferTexture2D = glfunction(glFramebufferTexture2D, 'iiiii')
        self.glFramebufferRenderbuffer = glfunction(glFramebufferRenderbuffer, 'iiii')
        self.glBlitFramebuffer = glfunction(glBlitFramebuffer, 'iiiiiiiiii')
        self.wglSwapIntervalEXT = glfunction(wglSwapIntervalEXT, 'i')

    def swap_interval(self, interval):
        self.wglSwapIntervalEXT(interval)

    def create_framebuffer(self, texture=0, renderbuffer=0):
        framebuffer = ctypes.c_int(0)
        self.glGenFramebuffers(1, ctypes.byref(framebuffer))
        self.glBindFramebuffer(0x8CA9, framebuffer)
        self.glBindFramebuffer(0x8CA8, framebuffer)
        if texture:
            self.glFramebufferTexture2D(0x8CA9, 0x8CE0, 0x0DE1, texture, 0)
        if renderbuffer:
            self.glFramebufferRenderbuffer(0x8CA9, 0x8CE0, 0x8D41, renderbuffer)
        draw_buffers = ctypes.c_int(0x8CE0)
        self.glDrawBuffers(1, ctypes.byref(draw_buffers))
        self.glReadBuffer(0x8CE0)
        self.glBindFramebuffer(0x8CA8, 0)
        self.glBindFramebuffer(0x8CA9, 0)
        return framebuffer

    def blit_framebuffer(self, framebuffer, size):
        width, height = size
        self.glBindFramebuffer(0x8CA8, framebuffer)
        self.glBindFramebuffer(0x8CA9, 0)
        self.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, 0x4000, 0x2600)
        self.glBindFramebuffer(0x8CA8, 0)
        self.glFlush()


def create_context(window):
    ctypes.windll.sdl2.SDL_GetWindowFromID.restype = ctypes.c_void_p
    ctypes.windll.sdl2.SDL_GL_CreateContext.argtypes = [ctypes.c_void_p]
    ctypes.windll.sdl2.SDL_GL_CreateContext.restype = ctypes.c_void_p
    ctypes.windll.sdl2.SDL_GL_MakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    win = ctypes.windll.sdl2.SDL_GetWindowFromID(window.id)
    ctx = ctypes.windll.sdl2.SDL_GL_CreateContext(win)
    ctypes.windll.sdl2.SDL_GL_MakeCurrent(win, ctx)
    return DetectContext()


def swap_window(window, ctx):
    ctypes.windll.sdl2.SDL_GetWindowFromID.restype = ctypes.c_void_p
    ctypes.windll.sdl2.SDL_GL_SwapWindow.argtypes = [ctypes.c_void_p]
    win = ctypes.windll.sdl2.SDL_GetWindowFromID(window.id)
    ctypes.windll.sdl2.SDL_GL_MakeCurrent(win, ctx)
    ctypes.windll.sdl2.SDL_GL_SwapWindow(win)


def dwm_flush():
    ctypes.windll.dwmapi.DwmFlush()


class GLWindow:
    def __init__(self, size, position, title):
        self.size = size
        restore = DetectContext()
        zengl_context.make_current()
        ctx = zengl.context()
        self.image = ctx.image(size, 'rgba8unorm', texture=False)
        self.window = pygame.Window(size=size, opengl=True)
        self.window.title = title
        self.window.position = position
        self.context = create_context(self.window)
        self.context.inherit(zengl_context)
        # self.framebuffer = gl.create_framebuffer(texture=zengl.inspect(self.image)['texture'])
        self.framebuffer = gl.create_framebuffer(renderbuffer=zengl.inspect(self.image)['renderbuffer'])
        gl.swap_interval(0)
        restore.make_current()

    def present(self):
        self.context.make_current()
        gl.blit_framebuffer(self.framebuffer, self.size)
        swap_window(self.window, self.context.hglrc)


def make_pipeline(uniform_buffer, texture, framebuffer):
    ctx = zengl.context()
    return ctx.pipeline(
        vertex_shader='''
            #version 300 es
            precision highp float;

            layout (std140) uniform Common {
                mat4 mvp;
                vec3 eye;
                vec3 light;
            };

            vec3 vertices[36] = vec3[](
                vec3(-0.5, -0.5, -0.5),
                vec3(-0.5, 0.5, -0.5),
                vec3(0.5, 0.5, -0.5),
                vec3(0.5, 0.5, -0.5),
                vec3(0.5, -0.5, -0.5),
                vec3(-0.5, -0.5, -0.5),
                vec3(-0.5, -0.5, 0.5),
                vec3(0.5, -0.5, 0.5),
                vec3(0.5, 0.5, 0.5),
                vec3(0.5, 0.5, 0.5),
                vec3(-0.5, 0.5, 0.5),
                vec3(-0.5, -0.5, 0.5),
                vec3(-0.5, -0.5, -0.5),
                vec3(0.5, -0.5, -0.5),
                vec3(0.5, -0.5, 0.5),
                vec3(0.5, -0.5, 0.5),
                vec3(-0.5, -0.5, 0.5),
                vec3(-0.5, -0.5, -0.5),
                vec3(0.5, -0.5, -0.5),
                vec3(0.5, 0.5, -0.5),
                vec3(0.5, 0.5, 0.5),
                vec3(0.5, 0.5, 0.5),
                vec3(0.5, -0.5, 0.5),
                vec3(0.5, -0.5, -0.5),
                vec3(0.5, 0.5, -0.5),
                vec3(-0.5, 0.5, -0.5),
                vec3(-0.5, 0.5, 0.5),
                vec3(-0.5, 0.5, 0.5),
                vec3(0.5, 0.5, 0.5),
                vec3(0.5, 0.5, -0.5),
                vec3(-0.5, 0.5, -0.5),
                vec3(-0.5, -0.5, -0.5),
                vec3(-0.5, -0.5, 0.5),
                vec3(-0.5, -0.5, 0.5),
                vec3(-0.5, 0.5, 0.5),
                vec3(-0.5, 0.5, -0.5)
            );

            vec3 normals[36] = vec3[](
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, -1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, 0.0, 1.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(0.0, -1.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0),
                vec3(-1.0, 0.0, 0.0)
            );

            vec2 texcoords[36] = vec2[](
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
                vec2(0.0, 0.0)
            );

            out vec3 v_vertex;
            out vec3 v_normal;
            out vec2 v_texcoord;

            void main() {
                v_vertex = vertices[gl_VertexID];
                v_normal = normals[gl_VertexID];
                v_texcoord = texcoords[gl_VertexID];
                gl_Position = mvp * vec4(v_vertex, 1.0);
            }
        ''',
        fragment_shader='''
            #version 300 es
            precision highp float;
            precision highp sampler2D;

            uniform sampler2D Texture;

            layout (std140) uniform Common {
                mat4 mvp;
                vec3 eye;
                vec3 light;
            };

            in vec3 v_vertex;
            in vec3 v_normal;
            in vec2 v_texcoord;

            layout (location = 0) out vec4 out_color;

            void main() {
                vec3 color = texture(Texture, v_texcoord).rgb;
                float lum = dot(normalize(light.xyz), normalize(v_normal)) * 0.7 + 0.3;
                out_color = vec4(pow(color * lum, vec3(1.0 / 2.2)), 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Common',
                'binding': 0,
            },
            {
                'name': 'Texture',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'uniform_buffer',
                'binding': 0,
                'buffer': uniform_buffer,
            },
            {
                'type': 'sampler',
                'binding': 0,
                'image': texture,
            },
        ],
        framebuffer=framebuffer,
        topology='triangles',
        cull_face='back',
        vertex_count=36,
    )


class UniformBuffer:
    def __init__(self):
        ctx = zengl.context()
        self.buffer = ctx.buffer(size=96, uniform=True)
        self.time = 0.0

    def update(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time * 0.4) * 3.0, math.sin(self.time * 0.4) * 3.0, 1.5)
        camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)
        light = eye[0], eye[1], eye[2] + 2.0
        self.buffer.write(struct.pack('64s3f4x3f4x', camera, *eye, *light))


class CubeRenderer:
    def __init__(self, uniform_buffer, texture, output):
        self.output = output
        self.depth = ctx.image(output.size, 'depth24plus', texture=False)
        self.pipeline = make_pipeline(uniform_buffer, texture, [output, self.depth])

    def render(self):
        self.output.clear()
        self.depth.clear()
        self.pipeline.render()


pygame.init()
zengl.init(zengl.loader(headless=True))

ctx = zengl.context()
uniform_buffer = UniformBuffer()

textures = [
    generate_texture((128, 128), 'Hi'),
    generate_texture((128, 128), 'Yo'),
    generate_texture((128, 128), ':D'),
    generate_texture((128, 128), ':)'),
]

zengl_context = DetectContext()
gl = DetectGL()

windows = [
    GLWindow((400, 400), (100, 100), 'Window 1'),
    GLWindow((400, 400), (600, 100), 'Window 2'),
    GLWindow((400, 400), (100, 600), 'Window 3'),
    GLWindow((400, 400), (600, 600), 'Window 4'),
]

renderers = [
    CubeRenderer(uniform_buffer.buffer, texture, output=window.image) for window, texture in zip(windows, textures)
]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    zengl_context.make_current()

    uniform_buffer.update()

    for renderer in renderers:
        renderer.render()

    for window in windows:
        window.present()

    dwm_flush()
