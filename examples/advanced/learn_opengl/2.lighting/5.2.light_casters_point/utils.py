import json
import os
import struct

import glm
import numpy as np
import pyglet
import requests
from progress.bar import Bar

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False

PYGLET_KEYS = {
    1: 'mouse1',
    2: 'mouse2',
    4: 'mouse3',
    32: 'space',
    48: '0',
    49: '1',
    50: '2',
    51: '3',
    52: '4',
    53: '5',
    54: '6',
    55: '7',
    56: '8',
    57: '9',
    97: 'a',
    98: 'b',
    99: 'c',
    100: 'd',
    101: 'e',
    102: 'f',
    103: 'g',
    104: 'h',
    105: 'i',
    106: 'j',
    107: 'k',
    108: 'l',
    109: 'm',
    110: 'n',
    111: 'o',
    112: 'p',
    113: 'q',
    114: 'r',
    115: 's',
    116: 't',
    117: 'u',
    118: 'v',
    119: 'w',
    120: 'x',
    121: 'y',
    122: 'z',
    65507: 'ctrl',
    65505: 'shift',
    65513: 'alt',
    65293: 'enter',
    65288: 'backspace',
    65307: 'escape',
    65289: 'tab',
    65535: 'delete',
    65361: 'left',
    65363: 'right',
    65362: 'up',
    65364: 'down',
    65365: 'pageup',
    65366: 'pagedown',
    65360: 'home',
    65367: 'end',
}

project_root = os.path.dirname(os.path.normpath(os.path.abspath(__file__)))

while not os.path.isfile(os.path.join(project_root, 'setup.py')):
    project_root = os.path.dirname(project_root)


def download(filename):
    os.makedirs(os.path.join(project_root, 'downloads'), exist_ok=True)
    full_path = os.path.join(project_root, 'downloads', filename)
    if os.path.isfile(full_path):
        return full_path
    with requests.get(f'https://f003.backblazeb2.com/file/zengl-data/examples/{filename}', stream=True) as request:
        if not request.ok:
            raise Exception(request.text)
        total_size = int(request.headers.get('Content-Length'))
        print(f'Downloading {filename}')
        bar = Bar('Progress', fill='-', suffix='%(percent)d%%', max=total_size)
        with open(full_path + '.temp', 'wb') as f:
            chunk_size = (total_size + 100 - 1) // 100
            for chunk in request.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.next(len(chunk))
        os.rename(full_path + '.temp', full_path)
        bar.finish()
    return full_path


class PygletWindow(pyglet.window.Window):
    def __init__(self, width, height):
        self.alive = True
        self.mouse = (0, 0)
        self.keys = set(), set()
        config = pyglet.gl.Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            double_buffer=False,
            depth_size=0,
            samples=0,
        )
        super().__init__(width=width, height=height, config=config, vsync=True)

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        pass

    def on_mouse_press(self, x, y, button, modifiers):
        self.keys[0].add(PYGLET_KEYS.get(button))

    def on_mouse_release(self, x, y, button, modifiers):
        self.keys[0].discard(PYGLET_KEYS.get(button))

    def on_key_press(self, symbol, modifiers):
        self.keys[0].add(PYGLET_KEYS.get(symbol))

    def on_key_release(self, symbol, modifiers):
        self.keys[0].discard(PYGLET_KEYS.get(symbol))

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (x, y)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.wheel += scroll_y

    def on_close(self):
        self.alive = False

    def update(self):
        self.wheel = 0.0
        self.keys = self.keys[0], self.keys[0].copy()
        self.flip()
        self.dispatch_events()
        return self.alive


class Window:
    def __init__(self, size=(1280, 720)):
        self._wnd = PygletWindow(*size)
        width, height = self._wnd.get_framebuffer_size()
        self.size = (width, height)
        self.aspect = width / height
        self.time = 0.0

    @property
    def mouse(self):
        return self._wnd.mouse

    @property
    def wheel(self):
        return self._wnd.wheel

    def key_pressed(self, key):
        return key in self._wnd.keys[0] and key not in self._wnd.keys[1]

    def key_released(self, key):
        return key not in self._wnd.keys[0] and key in self._wnd.keys[1]

    def key_down(self, key):
        return key in self._wnd.keys[0]

    def update(self):
        self.time += 1.0 / 60.0
        return self._wnd.update()


class Camera:
    def __init__(self, angle=0.0, tilt=0.0, distance=5.0):
        self.size = (1.0, 1.0)
        self.angle = angle
        self.tilt = tilt
        self.distance = distance
        self.fov = 45.0
        self.near = 0.1
        self.far = 100.0
        self.target = glm.vec3(0.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.start_drag = None

    @property
    def projection_matrix(self):
        return glm.perspective(glm.radians(self.fov), self.size[0] / self.size[1], self.near, self.far)

    @property
    def view_matrix(self):
        return glm.lookAt(self.position + self.target, self.target, self.up)

    @property
    def position(self):
        h, v = self.angle, self.tilt
        return glm.vec3(glm.cos(h) * glm.cos(v), glm.sin(v), glm.sin(h) * glm.cos(v)) * self.distance

    def update(self, window):
        if window.key_pressed('mouse1'):
            self.start_drag = window.mouse

        if window.key_down('mouse1'):
            dx, dy = glm.vec2(window.mouse) - glm.vec2(self.start_drag)
            self.angle = (self.angle + dx * 0.01) % (glm.pi() * 2.0)
            self.tilt = glm.clamp(self.tilt - dy * 0.01, -glm.pi() * 0.499, glm.pi() * 0.499)
            self.start_drag = window.mouse

        if window.key_released('mouse1'):
            self.start_drag = None

        self.distance *= 0.9 ** window.wheel
        self.size = window.size


def set_uniform_glm(pipeline, name, value):
    pipeline.uniforms[name][:] = value.to_bytes()


def set_uniform_float(pipeline, name, value):
    pipeline.uniforms[name][:] = struct.pack('f', value)


def set_uniform_int(pipeline, name, value):
    pipeline.uniforms[name][:] = struct.pack('i', value)


def read_file(filename, mode='r'):
    with open(os.path.join(os.path.abspath(__file__), '..', filename), mode) as f:
        return f.read()


def read_vertices(filename, dtype='f4'):
    return np.array(json.loads(read_file(filename)), dtype=dtype)
