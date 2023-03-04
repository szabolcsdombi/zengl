import importlib
import os

import imgui
from imgui.integrations.pyglet import create_renderer
from pyglet import gl

import assets
import window

examples = [
    ('zengl_logo', 'ZenGL Logo'),
    ('hello_triangle', 'Hello Triangle'),
    ('bezier_curves', 'Bezier Curves'),
    ('rigged_objects', 'Rigged Objects'),
    ('envmap', 'Environment Map'),
    ('normal_mapping', 'Normal Mapping'),
    ('grass', 'Grass'),
    ('box_grid', 'Box Grid'),
    ('blending', 'Blending'),
    ('julia_fractal', 'Fractal'),
    ('vertex_colors', 'Toon Shading'),
    ('deferred_rendering', 'Deferred Rendering'),
    ('crate', 'Crate'),
]

wnd = window.Window((1600, 900))
imgui.create_context()
imgui.get_io().ini_file_name = b''
impl = create_renderer(wnd._wnd)


class g:
    modules = {}
    example = None
    load_next = 'zengl_logo'
    download_progress = None


def load_example(name):
    if name in g.modules:
        importlib.reload(g.modules[name])
    else:
        g.modules[name] = importlib.import_module(name, '.')
    return g.modules[name]


def update(main_loop=True):
    if not main_loop and not g.load_next:
        if wnd.key_pressed('up'):
            index = next(i for i in range(-1, len(examples)) if examples[i + 1][0] == g.example)
            g.load_next = examples[index][0]
        if wnd.key_pressed('down'):
            index = next(i for i in range(len(examples)) if examples[i - 1][0] == g.example)
            g.load_next = examples[index][0]
    imgui.new_frame()
    # imgui.show_demo_window()
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (4.0, 6.0))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
    imgui.push_style_color(imgui.COLOR_TEXT_DISABLED, 1.0, 1.0, 1.0, 1.0)
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu('Examples', True):
            for example, title in examples:
                if imgui.menu_item(title, None, g.example == example, g.load_next is None)[0]:
                    g.load_next = example
            imgui.end_menu()
        if g.download_progress:
            filename, progress = g.download_progress
            full = int(progress * 50)
            line = 'Downloading |' + '-' * full + ' ' * (50 - full) + '| ' + filename
            if imgui.begin_menu(line, False):
                imgui.end_menu()
        if g.load_next:
            if imgui.begin_menu('Loading...', False):
                imgui.end_menu()
        imgui.end_main_menu_bar()
    imgui.pop_style_color()
    imgui.pop_style_var(2)
    imgui.render()
    impl.render(imgui.get_draw_data())
    res = wnd.update()
    if not main_loop and g.load_next:
        return False
    elif g.load_next:
        g.example = g.load_next
        g.load_next = None
        assets.Loader = LoaderHook
        window.Window = WindowHook
        wnd._wnd.set_caption(os.path.join(os.path.dirname(__file__), g.example + '.py'))
        module = load_example(g.example)
        module.ctx.release('all')
    return res


class LoaderHook:
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.index = 0

    def update(self, chunk_size):
        self.index += chunk_size
        g.download_progress = (self.filename, self.index / self.total_size)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        update(main_loop=False)

    def finish(self):
        g.download_progress = None


class WindowHook:
    def __init__(self):
        self.size = wnd.size
        self.aspect = wnd.aspect
        self.time = 0.0
        self.first = True

    @property
    def mouse(self):
        return wnd.mouse

    @property
    def wheel(self):
        return wnd.wheel

    def key_pressed(self, key):
        return wnd.key_pressed(key)

    def key_released(self, key):
        return wnd.key_released(key)

    def key_down(self, key):
        return wnd.key_down(key)

    def update(self):
        if self.first:
            self.first = False
            return True
        self.time += 1.0 / 60.0
        return update(main_loop=False)


while update():
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
