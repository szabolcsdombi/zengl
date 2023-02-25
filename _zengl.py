import re
import struct
import textwrap

FORMAT = {
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
    0x1404: (1, 'i'),
    0x8B53: (2, 'i'),
    0x8B54: (3, 'i'),
    0x8B55: (4, 'i'),
    0x8B56: (1, 'i'),
    0x8B57: (2, 'i'),
    0x8B58: (3, 'i'),
    0x8B59: (4, 'i'),
    0x1405: (1, 'I'),
    0x8DC6: (2, 'I'),
    0x8DC7: (3, 'I'),
    0x8DC8: (4, 'I'),
    0x1406: (1, 'f'),
    0x8B50: (2, 'f'),
    0x8B51: (3, 'f'),
    0x8B52: (4, 'f'),
    0x8B5A: (4, 'f'),
    0x8B65: (6, 'f'),
    0x8B66: (8, 'f'),
    0x8B67: (6, 'f'),
    0x8B5B: (9, 'f'),
    0x8B68: (12, 'f'),
    0x8B69: (8, 'f'),
    0x8B6A: (12, 'f'),
    0x8B5C: (16, 'f'),
}


def loader(headless=False):
    import glcontext
    mode = 'standalone' if headless else 'detect'
    return glcontext.default_backend()(glversion=330, mode=mode)


def calcsize(layout):
    nodes = layout.split(' ')
    if nodes[-1] == '/i':
        nodes.pop()
    stride = 0
    for node in nodes:
        if node[-1] == 'x':
            stride += int(node[:-1])
            continue
        stride += FORMAT[node][1]
    return stride


def bind(buffer, layout, *attributes):
    nodes = layout.split(' ')
    step = 'vertex'
    if nodes[-1] == '/i':
        step = 'instance'
        nodes.pop()
    res = []
    offset = 0
    idx = 0
    for node in nodes:
        if node[-1] == 'x':
            offset += int(node[:-1])
            continue
        if len(attributes) == idx:
            raise ValueError(f'Not enough vertex attributes for format "{layout}"')
        location = attributes[idx]
        format, size = FORMAT[node]
        if location >= 0:
            res.append({
                'location': location,
                'buffer': buffer,
                'format': format,
                'offset': offset,
                'step': step,
            })
        offset += size
        idx += 1

    if len(attributes) != idx:
        raise ValueError(f'Too many vertex attributes for format "{layout}"')

    for x in res:
        x['stride'] = offset

    return res


def vertex_array_bindings(vertex_buffers, index_buffer):
    res = [index_buffer]
    for obj in vertex_buffers:
        res.extend([obj['buffer'], obj['location'], obj['offset'], obj['stride'], STEP[obj['step']], obj['format']])
    return tuple(res)


def resource_bindings(resources):
    uniform_buffers = []
    for obj in sorted((x for x in resources if x['type'] == 'uniform_buffer'), key=lambda x: x['binding']):
        binding = obj['binding']
        buffer = obj['buffer']
        offset = obj.get('offset', 0)
        size = obj.get('size', buffer.size - offset)
        uniform_buffers.extend([binding, buffer, offset, size])

    storage_buffers = []
    for obj in sorted((x for x in resources if x['type'] == 'storage_buffer'), key=lambda x: x['binding']):
        binding = obj['binding']
        buffer = obj['buffer']
        offset = obj.get('offset', 0)
        size = obj.get('size', buffer.size - offset)
        storage_buffers.extend([binding, buffer, offset, size])

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

    images = []
    for obj in sorted((x for x in resources if x['type'] == 'image'), key=lambda x: x['binding']):
        images.extend([obj['binding'], obj['image']])

    return tuple(uniform_buffers), tuple(storage_buffers), tuple(samplers), tuple(images)


def framebuffer_attachments(size, attachments):
    if not attachments:
        if size is None:
            raise ValueError('Missing framebuffer')
        return size, (), None
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
    res = [len(attachments[1]), CULL_FACE[cull_face]]

    if depth is None:
        depth = {}

    if attachments[2] is not None and attachments[2].flags & 2:
        res.extend([True, COMPARE_FUNC[depth.get('func', 'less')], bool(depth.get('write', True))])

    else:
        res.append(False)

    if stencil is None:
        stencil = {}

    if attachments[2] is not None and attachments[2].flags & 4:
        front = stencil.get('front', stencil.get('both', {}))
        back = stencil.get('back', stencil.get('both', {}))
        res.extend([
            True,
            STENCIL_OP[front.get('fail_op', 'keep')],
            STENCIL_OP[front.get('pass_op', 'keep')],
            STENCIL_OP[front.get('depth_fail_op', 'keep')],
            COMPARE_FUNC[front.get('compare_op', 'always')],
            int(front.get('compare_mask', 0xff)),
            int(front.get('write_mask', 0xff)),
            int(front.get('reference', 0)),
            STENCIL_OP[back.get('fail_op', 'keep')],
            STENCIL_OP[back.get('pass_op', 'keep')],
            STENCIL_OP[back.get('depth_fail_op', 'keep')],
            COMPARE_FUNC[back.get('compare_op', 'always')],
            int(back.get('compare_mask', 0xff)),
            int(back.get('write_mask', 0xff)),
            int(back.get('reference', 0)),
        ])

    else:
        res.append(False)

    if blend is not None:
        res.append(True)
        for obj in blend:
            res.extend([
                int(obj.get('enable', True)),
                BLEND_FUNC[obj.get('op_color', 'add')],
                BLEND_FUNC[obj.get('op_alpha', 'add')],
                BLEND_CONSTANT[obj.get('src_color', 'one')],
                BLEND_CONSTANT[obj.get('dst_color', 'zero')],
                BLEND_CONSTANT[obj.get('src_alpha', 'one')],
                BLEND_CONSTANT[obj.get('dst_alpha', 'zero')],
            ])

    else:
        res.append(False)

    return tuple(res)


def program(includes, *shaders):
    def include(match):
        name = match.group(1)
        content = includes.get(name)
        if content is None:
            raise KeyError(f'cannot include "{name}"')
        return content

    res = []
    for shader, type in shaders:
        shader = textwrap.dedent(shader).strip()
        shader = re.sub(r'#include\s+"([^"]+)"', include, shader)
        shader = shader.encode().replace(b'\r', b'')
        res.append((shader, type))

    return tuple(res)


def compile_error(shader: bytes, shader_type: int, log: bytes):
    name = {0x8b31: 'Vertex Shader', 0x8b30: 'Fragment Shader', 0x91b9: 'Compute Shader'}[shader_type]
    log = log.rstrip(b'\x00').decode()
    raise ValueError(f'{name} Error\n\n{log}')


def linker_error(vertex_shader: bytes, fragment_shader: bytes, log: bytes):
    log = log.rstrip(b'\x00').decode()
    raise ValueError(f'Linker Error\n\n{log}')


def compute_linker_error(compute_shader: bytes, log: bytes):
    log = log.rstrip(b'\x00').decode()
    raise ValueError(f'Linker Error\n\n{log}')


def flatten(iterable):
    try:
        for x in iterable:
            yield from flatten(x)
    except TypeError:
        yield iterable


def clean_glsl_name(name):
    if name.endswith('[0]') and name['size'] > 1:
        return name[:-3]
    return name


def uniforms(interface, values):
    data = bytearray()
    uniform_map = {clean_glsl_name(obj['name']): obj for obj in interface if obj['type'] == 'uniform'}

    for name, value in values.items():
        if name not in uniform_map:
            raise KeyError(f'Uniform "{name}" does not exist')
        value = tuple(flatten(value))
        location = uniform_map[name]['location']
        size = uniform_map[name]['size']
        gltype = uniform_map[name]['gltype']
        if gltype not in UNIFORM_PACKER:
            raise ValueError(f'Uniform "{name}" has an unknown type')
        items, format = UNIFORM_PACKER[gltype]
        count = len(value) // items
        if len(value) > size * items:
            raise ValueError(f'Uniform "{name}" must be {size * items} long at most')
        if len(value) % items:
            raise ValueError(f'Uniform "{name}" must have a length divisible by {items}')
        data.extend(struct.pack('4i', len(value), location, count, gltype))
        for value in flatten(value):
            data.extend(struct.pack(format, value))
    data.extend(struct.pack('4i', 0, 0, 0, 0))
    return list(values), data


def validate(interface, resources, vertex_buffers, attachments, limits):
    errors = []

    unique = set((obj['type'], obj['binding']) for obj in resources)
    if len(resources) != len(unique):
        for obj in resources:
            key = (obj['type'], obj['binding'])
            if key not in unique:
                binding = obj['binding']
                rtype = obj['type']
                errors.append(f'Duplicate resource entry for "{rtype}" with binding = {binding}')
            unique.discard(key)

    unique = set(obj['location'] for obj in vertex_buffers)
    if len(vertex_buffers) != len(unique):
        for obj in vertex_buffers:
            location = obj['location']
            if location not in unique:
                errors.append(f'Duplicate vertex attribute entry with location = {location}')
            unique.discard(location)

    expected = set(obj['location'] + i for obj in interface if obj['type'] == 'input' for i in range(obj['size']))
    provided = set(obj['location'] for obj in vertex_buffers)

    if expected ^ provided:
        missing = expected - provided
        extra = provided - expected
        if missing:
            for location in sorted(missing):
                obj = next(obj for obj in interface if obj['type'] == 'input' and obj['location'] == location)
                name = clean_glsl_name(obj['name'])
                errors.append(f'Missing vertex buffer binding for "{name}" with location = {location}')
        if extra:
            for location in sorted(extra):
                errors.append(f'Unknown vertex attribute with location = {location}')

    expected = set(obj['location'] + i for obj in interface if obj['type'] == 'output' for i in range(obj['size']))
    provided = set(range(len(attachments)))
    if expected ^ provided:
        missing = expected - provided
        extra = provided - expected
        if missing:
            for location in sorted(missing):
                obj = next(obj for obj in interface if obj['type'] == 'output' and obj['location'] == location)
                name = clean_glsl_name(obj['name'])
                errors.append(f'Missing framebuffer attachment for "{name}" with location = {location}')
        if extra:
            for location in sorted(extra):
                errors.append(f'Unknown framebuffer attachment with location = {location}')

    expected = set(obj['binding'] for obj in interface if obj['type'] == 'uniform_buffer')
    provided = set(obj['binding'] for obj in resources if obj['type'] == 'uniform_buffer')
    if expected ^ provided:
        missing = expected - provided
        extra = provided - expected
        if missing:
            for binding in sorted(missing):
                obj = next(obj for obj in interface if obj['type'] == 'uniform_buffer' and obj['binding'] == binding)
                name = clean_glsl_name(obj['name'])
                errors.append(f'Missing uniform buffer binding for "{name}" with binding = {binding}')
        if extra:
            for binding in sorted(extra):
                errors.append(f'Unknown uniform buffer with binding = {binding}')

    expected = set(obj['binding'] for obj in interface if obj['type'] == 'storage_buffer')
    provided = set(obj['binding'] for obj in resources if obj['type'] == 'storage_buffer')
    if expected ^ provided:
        missing = expected - provided
        extra = provided - expected
        if missing:
            for binding in sorted(missing):
                obj = next(obj for obj in interface if obj['type'] == 'storage_buffer' and obj['binding'] == binding)
                name = clean_glsl_name(obj['name'])
                errors.append(f'Missing storage buffer binding for "{name}" with binding = {binding}')
        if extra:
            for binding in sorted(extra):
                errors.append(f'Unknown storage buffer with binding = {binding}')

    expected = set(obj['binding'] + i for obj in interface if obj['type'] == 'sampler' for i in range(obj['size']))
    provided = set(obj['binding'] for obj in resources if obj['type'] == 'sampler')
    if expected ^ provided:
        missing = expected - provided
        extra = provided - expected
        if missing:
            for binding in sorted(missing):
                obj = next(obj for obj in interface if obj['type'] == 'sampler' and obj['binding'] == binding)
                name = clean_glsl_name(obj['name'])
                errors.append(f'Missing sampler binding for "{name}" with binding = {binding}')
        if extra:
            for binding in sorted(extra):
                errors.append(f'Unknown sampler with binding = {binding}')

    expected = set(obj['binding'] + i for obj in interface if obj['type'] == 'image' for i in range(obj['size']))
    provided = set(obj['binding'] for obj in resources if obj['type'] == 'image')
    if expected ^ provided:
        missing = expected - provided
        extra = provided - expected
        if missing:
            for binding in sorted(missing):
                obj = next(obj for obj in interface if obj['type'] == 'image' and obj['binding'] == binding)
                name = clean_glsl_name(obj['name'])
                errors.append(f'Missing image binding for "{name}" with binding = {binding}')
        if extra:
            for binding in sorted(extra):
                errors.append(f'Unknown image with binding = {binding}')

    if errors:
        raise ValueError('Program Validation Error\n\n' + '\n'.join(errors))
