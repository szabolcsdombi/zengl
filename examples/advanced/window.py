import pyglet

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


class PygletWindow(pyglet.window.Window):
    def __init__(self, width, height):
        self.alive = True
        self.mouse = (0, 0)
        self.keys = set(), set()
        self.wheel = 0.0
        config = pyglet.gl.Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            double_buffer=True,
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
