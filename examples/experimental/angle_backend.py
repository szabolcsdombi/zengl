# NOTE: Copy a 'libGLESv2.dll' file to the same directory as this script. (VSCode or Chrome has one next to their exe)
import ctypes
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()
pygame.init()
pygame.display.set_mode((800, 600))

ctypes.windll.user32.GetActiveWindow.restype = ctypes.c_void_p
ctypes.windll.user32.GetDC.argtypes = [ctypes.c_void_p]
ctypes.windll.user32.GetDC.restype = ctypes.c_void_p

hwnd = ctypes.windll.user32.GetActiveWindow()
hdc = ctypes.windll.user32.GetDC(hwnd)

egl = ctypes.WinDLL('./libGLESv2.dll')
egl.EGL_GetDisplay.argtypes = [ctypes.c_void_p]
egl.EGL_GetDisplay.restype = ctypes.c_void_p
egl.EGL_Initialize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
egl.EGL_ChooseConfig.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
egl.EGL_CreateWindowSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
egl.EGL_CreateWindowSurface.restype = ctypes.c_void_p
egl.EGL_CreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
egl.EGL_CreateContext.restype = ctypes.c_void_p
egl.EGL_MakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
egl.EGL_SwapBuffers.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

display = egl.EGL_GetDisplay(hdc)
major, minor = ctypes.c_int(), ctypes.c_int()
egl.EGL_Initialize(display, ctypes.byref(major), ctypes.byref(minor))

num_configs = ctypes.c_int()
attribs = (ctypes.c_int * 32)()
attribs[0] = 0x3038
config = ctypes.c_void_p()
egl.EGL_ChooseConfig(display, attribs, ctypes.byref(config), 1, ctypes.byref(num_configs))

attribs = (ctypes.c_int * 32)()
attribs[0] = 0x3038
surface = egl.EGL_CreateWindowSurface(display, config, hwnd, attribs)
egl.EGL_BindAPI(0x30A0)

attribs = (ctypes.c_int * 32)()
attribs[0] = 0x3098
attribs[1] = 3
attribs[2] = 0x30FB
attribs[3] = 0
attribs[4] = 0x3038
context = egl.EGL_CreateContext(display, config, 0, attribs)
egl.EGL_MakeCurrent(display, surface, surface, context)
egl.EGL_SwapBuffers(display, surface)

egl.glGetString.restype = ctypes.c_char_p
print(egl.glGetString(0x1F00))
print(egl.glGetString(0x1F01))
print(egl.glGetString(0x1F02))

@ctypes.WINFUNCTYPE(None, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p)
def glGetBufferSubData(target, offset, size, data):
    pass

class Loader:
    def load_opengl_function(self, name):
        if name == 'glGetBufferSubData':
            return ctypes.cast(glGetBufferSubData, ctypes.c_void_p).value
        return ctypes.cast(egl[name], ctypes.c_void_p).value

zengl.init(Loader())

ctx = zengl.context()
size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec2 vertices[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.866, -0.7),
            vec2(0.866, -0.7)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        out vec3 v_color;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
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
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    egl.EGL_SwapBuffers(display, surface)
