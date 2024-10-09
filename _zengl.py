import re
import struct
import sys
import textwrap

__version__ = '2.6.0'

VERTEX_FORMAT = {
    'uint8x2': (0x1401, 2, 0, 1),
    'uint8x4': (0x1401, 4, 0, 1),
    'sint8x2': (0x1400, 2, 0, 1),
    'sint8x4': (0x1400, 4, 0, 1),
    'unorm8x2': (0x1401, 2, 1, 0),
    'unorm8x4': (0x1401, 4, 1, 0),
    'snorm8x2': (0x1400, 2, 1, 0),
    'snorm8x4': (0x1400, 4, 1, 0),
    'uint16x2': (0x1403, 2, 0, 1),
    'uint16x4': (0x1403, 4, 0, 1),
    'sint16x2': (0x1402, 2, 0, 1),
    'sint16x4': (0x1402, 4, 0, 1),
    'unorm16x2': (0x1403, 2, 1, 0),
    'unorm16x4': (0x1403, 4, 1, 0),
    'snorm16x2': (0x1402, 2, 1, 0),
    'snorm16x4': (0x1402, 4, 1, 0),
    'float16x2': (0x140B, 2, 0, 0),
    'float16x4': (0x140B, 4, 0, 0),
    'float32': (0x1406, 1, 0, 0),
    'float32x2': (0x1406, 2, 0, 0),
    'float32x3': (0x1406, 3, 0, 0),
    'float32x4': (0x1406, 4, 0, 0),
    'uint32': (0x1405, 1, 0, 1),
    'uint32x2': (0x1405, 2, 0, 1),
    'uint32x3': (0x1405, 3, 0, 1),
    'uint32x4': (0x1405, 4, 0, 1),
    'sint32': (0x1404, 1, 0, 1),
    'sint32x2': (0x1404, 2, 0, 1),
    'sint32x3': (0x1404, 3, 0, 1),
    'sint32x4': (0x1404, 4, 0, 1),
}

IMAGE_FORMAT = {
    'r8unorm': (0x8229, 0x1903, 0x1401, 0x1800, 1, 1, 1, 1, 'f'),
    'rg8unorm': (0x822B, 0x8227, 0x1401, 0x1800, 2, 2, 1, 1, 'f'),
    'rgba8unorm': (0x8058, 0x1908, 0x1401, 0x1800, 4, 4, 1, 1, 'f'),
    'r8snorm': (0x8F94, 0x1903, 0x1401, 0x1800, 1, 1, 1, 1, 'f'),
    'rg8snorm': (0x8F95, 0x8227, 0x1401, 0x1800, 2, 2, 1, 1, 'f'),
    'rgba8snorm': (0x8F97, 0x1908, 0x1401, 0x1800, 4, 4, 1, 1, 'f'),
    'r8uint': (0x8232, 0x8D94, 0x1401, 0x1800, 1, 1, 1, 1, 'u'),
    'rg8uint': (0x8238, 0x8228, 0x1401, 0x1800, 2, 2, 1, 1, 'u'),
    'rgba8uint': (0x8D7C, 0x8D99, 0x1401, 0x1800, 4, 4, 1, 1, 'u'),
    'r16uint': (0x8234, 0x8D94, 0x1403, 0x1800, 1, 2, 1, 1, 'u'),
    'rg16uint': (0x823A, 0x8228, 0x1403, 0x1800, 2, 4, 1, 1, 'u'),
    'rgba16uint': (0x8D76, 0x8D99, 0x1403, 0x1800, 4, 8, 1, 1, 'u'),
    'r32uint': (0x8236, 0x8D94, 0x1405, 0x1800, 1, 4, 1, 1, 'u'),
    'rg32uint': (0x823C, 0x8228, 0x1405, 0x1800, 2, 8, 1, 1, 'u'),
    'rgba32uint': (0x8D70, 0x8D99, 0x1405, 0x1800, 4, 16, 1, 1, 'u'),
    'r8sint': (0x8231, 0x8D94, 0x1400, 0x1800, 1, 1, 1, 1, 'i'),
    'rg8sint': (0x8237, 0x8228, 0x1400, 0x1800, 2, 2, 1, 1, 'i'),
    'rgba8sint': (0x8D8E, 0x8D99, 0x1400, 0x1800, 4, 4, 1, 1, 'i'),
    'r16sint': (0x8233, 0x8D94, 0x1402, 0x1800, 1, 2, 1, 1, 'i'),
    'rg16sint': (0x8239, 0x8228, 0x1402, 0x1800, 2, 4, 1, 1, 'i'),
    'rgba16sint': (0x8D88, 0x8D99, 0x1402, 0x1800, 4, 8, 1, 1, 'i'),
    'r32sint': (0x8235, 0x8D94, 0x1404, 0x1800, 1, 4, 1, 1, 'i'),
    'rg32sint': (0x823B, 0x8228, 0x1404, 0x1800, 2, 8, 1, 1, 'i'),
    'rgba32sint': (0x8D82, 0x8D99, 0x1404, 0x1800, 4, 16, 1, 1, 'i'),
    'r16float': (0x822D, 0x1903, 0x1406, 0x1800, 1, 2, 1, 1, 'f'),
    'rg16float': (0x822F, 0x8227, 0x1406, 0x1800, 2, 4, 1, 1, 'f'),
    'rgba16float': (0x881A, 0x1908, 0x1406, 0x1800, 4, 8, 1, 1, 'f'),
    'r32float': (0x822E, 0x1903, 0x1406, 0x1800, 1, 4, 1, 1, 'f'),
    'rg32float': (0x8230, 0x8227, 0x1406, 0x1800, 2, 8, 1, 1, 'f'),
    'rgba32float': (0x8814, 0x1908, 0x1406, 0x1800, 4, 16, 1, 1, 'f'),
    'rgb10a2unorm': (0x8059, 0x1908, 0x8368, 0x1800, 4, 4, 1, 1, 'f'),
    'depth16unorm': (0x81A5, 0x1902, 0x1403, 0x1801, 1, 2, 0, 2, 'f'),
    'depth24plus': (0x81A6, 0x1902, 0x1405, 0x1801, 1, 4, 0, 2, 'f'),
    'depth24plus-stencil8': (0x88F0, 0x84F9, 0x84FA, 0x84F9, 2, 4, 0, 6, 'x'),
    'depth32float': (0x8CAC, 0x1902, 0x1406, 0x1801, 1, 4, 0, 2, 'f'),
}

TOPOLOGY = {
    'points': 0,
    'lines': 1,
    'line_loop': 2,
    'line_strip': 3,
    'triangles': 4,
    'triangle_strip': 5,
    'triangle_fan': 6,
}

SHORT_VERTEX_FORMAT = {
    '2u1': ('uint8x2', 2),
    '4u1': ('uint8x4', 4),
    '2i1': ('sint8x2', 2),
    '4i1': ('sint8x4', 4),
    '2nu1': ('unorm8x2', 2),
    '4nu1': ('unorm8x4', 4),
    '2ni1': ('snorm8x2', 2),
    '4ni1': ('snorm8x4', 4),
    '2u2': ('uint16x2', 4),
    '4u2': ('uint16x4', 8),
    '2i2': ('sint16x2', 4),
    '4i2': ('sint16x4', 8),
    '2nu2': ('unorm16x2', 4),
    '4nu2': ('unorm16x4', 8),
    '2ni2': ('snorm16x2', 4),
    '4ni2': ('snorm16x4', 8),
    '2h': ('float16x2', 4),
    '4h': ('float16x4', 8),
    '1f': ('float32', 4),
    '2f': ('float32x2', 8),
    '3f': ('float32x3', 12),
    '4f': ('float32x4', 16),
    '1u': ('uint32', 4),
    '2u': ('uint32x2', 8),
    '3u': ('uint32x3', 12),
    '4u': ('uint32x4', 16),
    '1i': ('sint32', 4),
    '2i': ('sint32x2', 8),
    '3i': ('sint32x3', 12),
    '4i': ('sint32x4', 16),
}

BUFFER_ACCESS = {
    'stream_draw': 0x88E0,
    'stream_read': 0x88E1,
    'stream_copy': 0x88E2,
    'static_draw': 0x88E4,
    'static_read': 0x88E5,
    'static_copy': 0x88E6,
    'dynamic_draw': 0x88E8,
    'dynamic_read': 0x88E9,
    'dynamic_copy': 0x88EA,
}

CULL_FACE = {
    'front': 0x0404,
    'back': 0x0405,
    'front_and_back': 0x0408,
    'none': 0,
}

MIN_FILTER = {
    'nearest': 0x2600,
    'linear': 0x2601,
    'nearest_mipmap_nearest': 0x2700,
    'linear_mipmap_nearest': 0x2701,
    'nearest_mipmap_linear': 0x2702,
    'linear_mipmap_linear': 0x2703,
}

MAG_FILTER = {
    'nearest': 0x2600,
    'linear': 0x2601,
}

TEXTURE_WRAP = {
    'repeat': 0x2901,
    'clamp_to_edge': 0x812F,
    'mirrored_repeat': 0x8370,
}

COMPARE_MODE = {
    'ref_to_texture': 0x884E,
    'none': 0,
}

COMPARE_FUNC = {
    'never': 0x0200,
    'less': 0x0201,
    'equal': 0x0202,
    'lequal': 0x0203,
    'greater': 0x0204,
    'notequal': 0x0205,
    'gequal': 0x0206,
    'always': 0x0207,
}

BLEND_FUNC = {
    'add': 0x8006,
    'subtract': 0x800A,
    'reverse_subtract': 0x800B,
    'min': 0x8007,
    'max': 0x8008,
}

BLEND_CONSTANT = {
    'zero': 0,
    'one': 1,
    'src_color': 0x0300,
    'one_minus_src_color': 0x0301,
    'src_alpha': 0x0302,
    'one_minus_src_alpha': 0x0303,
    'dst_alpha': 0x0304,
    'one_minus_dst_alpha': 0x0305,
    'dst_color': 0x0306,
    'one_minus_dst_color': 0x0307,
    'src_alpha_saturate': 0x0308,
    'constant_color': 0x8001,
    'one_minus_constant_color': 0x8002,
    'constant_alpha': 0x8003,
    'one_minus_constant_alpha': 0x8004,
    'src1_alpha': 0x8589,
    'src1_color': 0x88F9,
    'one_minus_src1_color': 0x88FA,
    'one_minus_src1_alpha': 0x88FB,
}

STENCIL_OP = {
    'zero': 0,
    'keep': 0x1E00,
    'replace': 0x1E01,
    'incr': 0x1E02,
    'decr': 0x1E03,
    'invert': 0x150A,
    'incr_wrap': 0x8507,
    'decr_wrap': 0x8508,
}

STEP = {
    'vertex': 0,
    'instance': 1,
}

VERTEX_SHADER_BUILTINS = {
    'gl_VertexID',
    'gl_InstanceID',
    'gl_DrawID',
    'gl_BaseVertex',
    'gl_BaseInstance',
}

UNIFORM_PACKER = {
    0x1404: (0, 1, 'i'),
    0x8B53: (1, 2, 'i'),
    0x8B54: (2, 3, 'i'),
    0x8B55: (3, 4, 'i'),
    0x8B56: (4, 1, 'i'),
    0x8B57: (5, 2, 'i'),
    0x8B58: (6, 3, 'i'),
    0x8B59: (7, 4, 'i'),
    0x1405: (8, 1, 'I'),
    0x8DC6: (9, 2, 'I'),
    0x8DC7: (10, 3, 'I'),
    0x8DC8: (11, 4, 'I'),
    0x1406: (12, 1, 'f'),
    0x8B50: (13, 2, 'f'),
    0x8B51: (14, 3, 'f'),
    0x8B52: (15, 4, 'f'),
    0x8B5A: (16, 4, 'f'),
    0x8B65: (17, 6, 'f'),
    0x8B66: (18, 8, 'f'),
    0x8B67: (19, 6, 'f'),
    0x8B5B: (20, 9, 'f'),
    0x8B68: (21, 12, 'f'),
    0x8B69: (22, 8, 'f'),
    0x8B6A: (23, 12, 'f'),
    0x8B5C: (24, 16, 'f'),
}


class DefaultLoader:
    def __init__(self):
        import ctypes

        def funcptr(lib, name):
            return ctypes.cast(getattr(lib, name, 0), ctypes.c_void_p).value or 0

        if sys.platform.startswith('win'):
            lib = ctypes.WinDLL('opengl32.dll')
            proc = ctypes.cast(lib.wglGetProcAddress, ctypes.WINFUNCTYPE(ctypes.c_ulonglong, ctypes.c_char_p))
            if not lib.wglGetCurrentContext():
                raise RuntimeError('Cannot detect window with OpenGL support')

            def loader(name):
                return proc(name.encode()) or funcptr(lib, name)

        elif sys.platform.startswith('linux'):
            try:
                lib = ctypes.CDLL('libEGL.so')
                proc = ctypes.cast(lib.eglGetProcAddress, ctypes.CFUNCTYPE(ctypes.c_ulonglong, ctypes.c_char_p))
                if not lib.eglGetCurrentContext():
                    raise RuntimeError('Cannot detect window with OpenGL support')

                def loader(name):
                    return proc(name.encode())

            except:
                lib = ctypes.CDLL('libGL.so')
                proc = ctypes.cast(lib.glXGetProcAddress, ctypes.CFUNCTYPE(ctypes.c_ulonglong, ctypes.c_char_p))
                if not lib.glXGetCurrentContext():
                    raise RuntimeError('Cannot detect window with OpenGL support') from None

                def loader(name):
                    return proc(name.encode()) or funcptr(lib, name)

        elif sys.platform.startswith('darwin'):
            lib = ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')

            def loader(name):
                return funcptr(lib, name)

        elif sys.platform.startswith('emscripten'):
            lib = ctypes.CDLL(None)

            def loader(name):
                return funcptr(lib, name)

        elif sys.platform.startswith('wasi'):
            lib = ctypes.CDLL(None)

            def loader(name):
                return funcptr(lib, name)

        self.load_opengl_function = loader


def headless_context_windows():
    from ctypes import c_int, c_void_p, cast, create_string_buffer, windll
    GetModuleHandle = windll.kernel32.GetModuleHandleA
    GetModuleHandle.restype = c_void_p
    RegisterClass = windll.user32.RegisterClassA
    RegisterClass.argtypes = [c_void_p]
    CreateWindow = windll.user32.CreateWindowExA
    CreateWindow.argtypes = [c_int] + [c_void_p] * 2 + [c_int] * 5 + [c_void_p] * 4
    CreateWindow.restype = c_void_p
    GetDC = windll.user32.GetDC
    GetDC.argtypes = [c_void_p]
    GetDC.restype = c_void_p
    DescribePixelFormat = windll.gdi32.DescribePixelFormat
    DescribePixelFormat.argtypes = [c_void_p, c_int, c_int, c_void_p]
    SetPixelFormat = windll.gdi32.SetPixelFormat
    SetPixelFormat.argtypes = [c_void_p, c_int, c_void_p]
    wglCreateContext = windll.opengl32.wglCreateContext
    wglCreateContext.argtypes = [c_void_p]
    wglCreateContext.restype = c_void_p
    wglMakeCurrent = windll.opengl32.wglMakeCurrent
    wglMakeCurrent.argtypes = [c_void_p, c_void_p]
    DefWindowProc = cast(windll.user32.DefWindowProcA, c_void_p).value
    hinstance = GetModuleHandle(0)
    classname = cast(b'glwindow', c_void_p).value
    wndclass = struct.pack('IQ8xQ32xQ', 32, DefWindowProc, hinstance, classname)
    RegisterClass(wndclass)
    hwnd = CreateWindow(0, classname, 0, 0, 0, 0, 0, 0, 0, 0, hinstance, 0)
    hdc = GetDC(hwnd)
    pfd = create_string_buffer(40)
    DescribePixelFormat(hdc, 1, 40, pfd)
    SetPixelFormat(hdc, 1, pfd)
    hglrc = wglCreateContext(hdc)
    wglMakeCurrent(hdc, hglrc)
    return hwnd, hdc, hglrc


def headless_context_glcontext():
    import glcontext
    return glcontext.default_backend()(glversion=330, mode='standalone')


def web_context():
    import js
    import zengl

    try:
        import pyodide_js
        module = pyodide_js._module

    except ImportError:
        module = js.window

    canvas = js.document.getElementById('canvas')

    if canvas is None:
        canvas = js.document.createElement('canvas')
        canvas.id = 'canvas'
        canvas.style.position = 'fixed'
        canvas.style.top = '0'
        canvas.style.right = '0'
        canvas.style.zIndex = '10'
        js.document.body.appendChild(canvas)

    options = js.Object()
    options.powerPreference = 'high-performance'
    options.premultipliedAlpha = False
    options.antialias = False
    options.alpha = False
    options.depth = False
    options.stencil = False

    gl = canvas.getContext('webgl2', options)
    callback = js.window.eval(zengl._extern_gl)
    symbols = callback(module, gl)
    module.mergeLibSymbols(symbols)
    return canvas, gl, symbols


def loader(headless=False):
    extra = None

    if headless:
        if sys.platform.startswith('win'):
            extra = headless_context_windows()

        else:
            extra = headless_context_glcontext()

    if sys.platform.startswith('emscripten'):
        extra = web_context()

    loader = DefaultLoader()
    loader.extra = extra
    return loader


def calcsize(layout):
    nodes = layout.split(' ')
    if nodes[-1] == '/i':
        nodes.pop()
    stride = 0
    for node in nodes:
        if node[-1] == 'x':
            stride += int(node[:-1])
            continue
        stride += SHORT_VERTEX_FORMAT[node][1]
    return stride


def bind(buffer, layout, *attributes, offset=0, instance=False):
    nodes = layout.split(' ')
    step = 'instance' if instance else 'vertex'
    if nodes[-1] == '/i':
        step = 'instance'
        nodes.pop()
    res = []
    sub_offset = 0
    idx = 0
    for node in nodes:
        if node[-1] == 'x':
            sub_offset += int(node[:-1])
            continue
        if len(attributes) == idx:
            raise ValueError(f'Not enough vertex attributes for format "{layout}"')
        location = attributes[idx]
        format, size = SHORT_VERTEX_FORMAT[node]
        if location >= 0:
            res.append(
                {
                    'location': location,
                    'buffer': buffer,
                    'format': format,
                    'offset': offset + sub_offset,
                    'step': step,
                }
            )
        sub_offset += size
        idx += 1

    if len(attributes) != idx:
        raise ValueError(f'Too many vertex attributes for format "{layout}"')

    for x in res:
        x['stride'] = sub_offset

    return res


def vertex_array_bindings(vertex_buffers, index_buffer):
    res = [index_buffer]
    for obj in vertex_buffers:
        buffer = obj['buffer']
        if buffer is not None:
            res.extend([buffer, obj['location'], obj['offset'], obj['stride'], STEP[obj['step']], obj['format']])
    return tuple(res)


def resource_bindings(resources):
    uniform_buffers = []
    for obj in sorted((x for x in resources if x['type'] == 'uniform_buffer'), key=lambda x: x['binding']):
        binding = obj['binding']
        buffer = obj['buffer']
        offset = obj.get('offset', 0)
        size = obj.get('size', buffer.size - offset)
        uniform_buffers.extend([binding, buffer, offset, size])

    samplers = []
    for obj in sorted((x for x in resources if x['type'] == 'sampler'), key=lambda x: x['binding']):
        params = (
            MIN_FILTER[obj.get('min_filter', 'linear')],
            MAG_FILTER[obj.get('mag_filter', 'linear')],
            float(obj.get('min_lod', -1000.0)),
            float(obj.get('max_lod', 1000.0)),
            float(obj.get('lod_bias', 0.0)),
            TEXTURE_WRAP[obj.get('wrap_x', 'repeat')],
            TEXTURE_WRAP[obj.get('wrap_y', 'repeat')],
            TEXTURE_WRAP[obj.get('wrap_z', 'repeat')],
            COMPARE_MODE[obj.get('compare_mode', 'none')],
            COMPARE_FUNC[obj.get('compare_func', 'never')],
            float(obj.get('max_anisotropy', 1.0)),
        )
        samplers.extend([obj['binding'], obj['image'], params])

    return tuple(uniform_buffers), tuple(samplers)


def framebuffer_attachments(attachments):
    if attachments is None:
        return None
    attachments = [x.face() if hasattr(x, 'face') else x for x in attachments]
    size = attachments[0].size
    samples = attachments[0].samples
    for attachment in attachments:
        if attachment.size != size:
            raise ValueError('Attachments must be images with the same size')
        if attachment.samples != samples:
            raise ValueError('Attachments must be images with the same number of samples')
    depth_stencil_attachment = None
    if not attachments[-1].flags & 1:
        depth_stencil_attachment = attachments[-1]
        attachments = attachments[:-1]
    for attachment in attachments:
        if not attachment.flags & 1:
            raise ValueError('The depth stencil attachments must be the last item in the framebuffer')
    return size, tuple(attachments), depth_stencil_attachment


def settings(cull_face, depth, stencil, blend, attachments):
    if attachments:
        num_color_attachments = len(attachments[1])
        has_depth = attachments[2] is not None and attachments[2].flags & 2
        has_stencil = attachments[2] is not None and attachments[2].flags & 4
    else:
        num_color_attachments = 1
        has_depth = False
        has_stencil = False

    res = [num_color_attachments, CULL_FACE[cull_face]]

    if depth is None:
        depth = {}

    if has_depth:
        res.extend([True, COMPARE_FUNC[depth.get('func', 'less')], bool(depth.get('write', True))])

    else:
        res.append(False)

    if stencil is None:
        stencil = {}

    if has_stencil:
        front = stencil.get('front', stencil.get('both', {}))
        back = stencil.get('back', stencil.get('both', {}))
        res.extend(
            [
                True,
                STENCIL_OP[front.get('fail_op', 'keep')],
                STENCIL_OP[front.get('pass_op', 'keep')],
                STENCIL_OP[front.get('depth_fail_op', 'keep')],
                COMPARE_FUNC[front.get('compare_op', 'always')],
                int(front.get('compare_mask', 0xFF)),
                int(front.get('write_mask', 0xFF)),
                int(front.get('reference', 0)),
                STENCIL_OP[back.get('fail_op', 'keep')],
                STENCIL_OP[back.get('pass_op', 'keep')],
                STENCIL_OP[back.get('depth_fail_op', 'keep')],
                COMPARE_FUNC[back.get('compare_op', 'always')],
                int(back.get('compare_mask', 0xFF)),
                int(back.get('write_mask', 0xFF)),
                int(back.get('reference', 0)),
            ]
        )

    else:
        res.append(False)

    if blend is not None:
        res.append(True)
        res.extend(
            [
                BLEND_FUNC[blend.get('op_color', 'add')],
                BLEND_FUNC[blend.get('op_alpha', 'add')],
                BLEND_CONSTANT[blend.get('src_color', 'one')],
                BLEND_CONSTANT[blend.get('dst_color', 'zero')],
                BLEND_CONSTANT[blend.get('src_alpha', 'one')],
                BLEND_CONSTANT[blend.get('dst_alpha', 'zero')],
            ]
        )

    else:
        res.append(False)

    return tuple(res)


def shader_source(source: str) -> bytes:
    return source.encode()


def program(vertex_shader, fragment_shader, layout, includes):
    def include(match):
        name = match.group(1)
        content = includes.get(name)
        if content is None:
            raise KeyError(f'cannot include "{name}"')
        return content

    vert = textwrap.dedent(vertex_shader).strip()
    vert = re.sub(r'#include\s+[<"]([^">]*)[">]', include, vert)
    vert = shader_source(vert)

    frag = textwrap.dedent(fragment_shader).strip()
    frag = re.sub(r'#include\s+[<"]([^">]*)[">]', include, frag)
    frag = shader_source(frag)

    bindings = []
    for obj in sorted(layout, key=lambda x: x['name']):
        bindings.extend((obj['name'], obj['binding']))

    return (vert, 0x8B31), (frag, 0x8B30), tuple(bindings)


def compile_error(shader: bytes, shader_type: int, log: bytes):
    name = {0x8B31: 'Vertex Shader', 0x8B30: 'Fragment Shader'}[shader_type]
    log = log.rstrip(b'\x00').decode(errors='ignore')
    raise ValueError(f'{name} Error\n\n{log}')


def linker_error(vertex_shader: bytes, fragment_shader: bytes, log: bytes):
    log = log.rstrip(b'\x00').decode(errors='ignore')
    raise ValueError(f'Linker Error\n\n{log}')


def flatten(iterable):
    try:
        for x in iterable:
            yield from flatten(x)
    except TypeError:
        yield iterable


def clean_glsl_name(name):
    if name.endswith('[0]'):
        return name[:-3]
    return name


def uniforms(interface, selection, uniform_data):
    uniform_map = {clean_glsl_name(obj['name']): obj for obj in interface[1]}
    uniforms = []
    layout = bytearray()
    offset = 0

    layout.extend(struct.pack('i', len(selection)))
    for name, values in selection.items():
        if name not in uniform_map:
            raise KeyError(f'Uniform "{name}" does not exist')
        location = uniform_map[name]['location']
        size = uniform_map[name]['size']
        gltype = uniform_map[name]['gltype']
        if gltype not in UNIFORM_PACKER:
            raise ValueError(f'Uniform "{name}" has an unknown type')
        function, items, format = UNIFORM_PACKER[gltype]
        if values is None:
            values_count = size * items
            values = bytes(values_count * 4)
        else:
            values = tuple(flatten(values))
            values_count = len(values)
            values = b''.join(struct.pack(format, x) for x in values)
        count = values_count // items
        if values_count > size * items:
            raise ValueError(f'Uniform "{name}" must be {size * items} long at most')
        if values_count % items:
            raise ValueError(f'Uniform "{name}" must have a length divisible by {items}')
        layout.extend(struct.pack('4i', function, location, count, offset))
        uniforms.append((name, slice(offset, offset + len(values)), values))
        offset += len(values)

    data = uniform_data if uniform_data else memoryview(bytearray(offset))

    if len(data) != offset:
        raise ValueError(f'uniform_data must be {offset} bytes long')

    mapping = {}
    for name, idx, values in uniforms:
        data[idx] = values
        mapping[name] = data[idx]
    return mapping, memoryview(layout), data


def layout_bindings(layout):
    res = []
    if not layout:
        return res
    for obj in layout:
        name = str(obj['name'])
        binding = int(obj['binding'])
        res.append((name, binding))
    return res


def validate(interface, layout, resources, vertex_buffers, info):
    attributes, uniforms, uniform_buffers = interface
    attributes = [
        {
            'name': obj['name'].replace('[0]', f'[{i:d}]'),
            'location': obj['location'] + i if obj['location'] >= 0 else -1,
        }
        for obj in attributes
        for i in range(obj['size'])
        if obj['name'] not in VERTEX_SHADER_BUILTINS
    ]
    uniforms = [
        {
            'name': obj['name'].replace('[0]', f'[{i}]'),
            'location': obj['location'] + i if obj['location'] >= 0 else -1,
        }
        for obj in uniforms
        for i in range(obj['size'])
        if obj['gltype'] not in UNIFORM_PACKER
    ]
    bound_attributes = set()
    bound_uniforms = set()
    bound_uniform_buffers = set()
    uniform_binding_map = {}
    uniform_buffer_binding_map = {}
    attribute_map = {obj['location']: obj for obj in attributes}
    uniform_map = {obj['name']: obj for obj in uniforms}
    uniform_buffer_map = {obj['name']: obj for obj in uniform_buffers}
    layout_map = {obj['name']: obj for obj in layout}
    uniform_buffer_resources = {obj['binding']: obj for obj in resources if obj['type'] == 'uniform_buffer'}
    sampler_resources = {obj['binding']: obj for obj in resources if obj['type'] == 'sampler'}
    max_uniform_block_size = info['max_uniform_block_size']

    for obj in uniform_buffers:
        if obj['size'] > max_uniform_block_size:
            name = obj['name']
            reason = f'the maximum supported size is {max_uniform_block_size}'
            raise ValueError(f'Uniform buffer "{name}" is too large, {reason}')

    for obj in vertex_buffers:
        location = obj['location']
        if location < 0:
            continue
        if location not in attribute_map:
            raise ValueError(f'Invalid vertex attribute location {location}')
        if location in bound_attributes:
            name = attribute_map[location]['name']
            raise ValueError(f'Duplicate vertex attribute binding for "{name}" at location {location}')
        bound_attributes.add(location)

    for obj in attributes:
        location = obj['location']
        if location < 0:
            continue
        if location not in bound_attributes:
            name = obj['name']
            raise ValueError(f'Unbound vertex attribute "{name}" at location {location}')

    for obj in layout:
        name = obj['name']
        binding = obj['binding']
        if name in uniform_map:
            uniform_binding_map[binding] = obj
            if binding < 0 or binding >= info['max_combined_texture_image_units']:
                raise ValueError(f'Invalid sampler binding for "{name}" with binding {binding}')
        elif name in uniform_buffer_map:
            uniform_buffer_binding_map[binding] = obj
            if binding < 0 or binding >= info['max_uniform_buffer_bindings']:
                raise ValueError(f'Invalid uniform buffer binding for "{name}" with binding {binding}')
        else:
            raise ValueError(f'Cannot set layout binding for "{name}"')

    for obj in uniforms:
        name = obj['name']
        location = obj['location']
        if location < 0:
            continue
        if name not in layout_map:
            raise ValueError(f'Missing layout binding for "{name}"')
        binding = layout_map[name]['binding']
        if binding not in sampler_resources:
            raise ValueError(f'Missing resource for "{name}" with binding {binding}')

    for obj in uniform_buffers:
        name = obj['name']
        if name not in layout_map:
            raise ValueError(f'Missing layout binding for "{name}"')
        binding = layout_map[name]['binding']
        if binding not in uniform_buffer_resources:
            raise ValueError(f'Missing resource for "{name}" with binding {binding}')

    for obj in resources:
        resource_type = obj['type']
        binding = obj['binding']
        if resource_type == 'uniform_buffer':
            buffer = obj['buffer']
            if binding not in uniform_buffer_binding_map:
                raise ValueError(f'Uniform buffer binding {binding} does not exist')
            name = uniform_buffer_binding_map[binding]['name']
            if binding in bound_uniform_buffers:
                raise ValueError(f'Duplicate uniform buffer binding for "{name}" with binding {binding}')
            size = uniform_buffer_map[name]['size']
            if buffer.size < size:
                reason = f'{buffer.size} is less than {size} for "{name}" with binding {binding}'
                raise ValueError(f'Uniform buffer is too small, {reason}')
            bound_uniform_buffers.add(binding)
        elif resource_type == 'sampler':
            image = obj['image']
            if binding not in uniform_binding_map:
                raise ValueError(f'Sampler binding {binding} does not exist')
            name = uniform_binding_map[binding]['name']
            if binding in bound_uniforms:
                raise ValueError(f'Duplicate sampler binding for "{name}" with binding {binding}')
            if image.renderbuffer:
                raise ValueError(f'Renderbuffers cannot be attached to "{name}" with binding {binding}')
            if image.samples != 1:
                raise ValueError(f'Multisample images cannot be attached to "{name}" with binding {binding}')
            bound_uniforms.add(binding)
        else:
            raise ValueError(f'Invalid resource type "{resource_type}"')
