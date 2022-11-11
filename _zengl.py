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


def buffer_bindings(resources):
    res = []
    for obj in sorted((x for x in resources if x['type'] == 'uniform_buffer'), key=lambda x: x['binding']):
        binding = obj['binding']
        buffer = obj['buffer']
        offset = obj.get('offset', 0)
        size = obj.get('size', buffer.size - offset)
        res.extend([binding, buffer, offset, size])
    return tuple(res)


def sampler_bindings(resources):
    res = []
    for obj in sorted((x for x in resources if x['type'] == 'sampler'), key=lambda x: x['binding']):
        border_color = obj.get('border_color', (0.0, 0.0, 0.0, 0.0))
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
            float(border_color[0]),
            float(border_color[1]),
            float(border_color[2]),
            float(border_color[3]),
        )
        res.extend([obj['binding'], obj['image'], params])
    return tuple(res)


def framebuffer_attachments(attachments):
    attachments = [x.face() if hasattr(x, 'face') else x for x in attachments]
    size = attachments[0].size
    samples = attachments[0].samples
    for attachment in attachments:
        if attachment.size != size:
            raise ValueError('Attachments must be images with the same size')
        if attachment.samples != samples:
            raise ValueError('Attachments must be images with the same number of samples')
    depth_stencil_attachment = None
    if not attachments[-1].color:
        depth_stencil_attachment = attachments[-1]
        attachments = attachments[:-1]
    for attachment in attachments:
        if not attachment.color:
            raise ValueError('The depth stencil attachments must be the last item in the framebuffer')
    return size, tuple(attachments), depth_stencil_attachment


def settings(primitive_restart, cull_face, color_mask, depth, stencil, blending, polygon_offset, attachments):
    res = [bool(primitive_restart), CULL_FACE[cull_face], color_mask]

    if depth is True or depth is False:
        res.extend([depth, depth, 0x0201])

    else:
        res.extend([bool(depth['test']), bool(depth['write']), COMPARE_FUNC[depth.get('func', 'less')]])

    if stencil is False:
        res.extend([
            False, 0x1E00, 0x1E00, 0x1E00, 0x0207, 0xff, 0xff, 0, 0x1E00, 0x1E00, 0x1E00, 0x0207, 0xff, 0xff, 0,
        ])

    else:
        front = stencil.get('front', stencil.get('both', {}))
        back = stencil.get('back', stencil.get('both', {}))
        res.extend([
            bool(stencil['test']),
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

    if blending is False:
        res.extend([0, 0x8006, 0x8006, 1, 0, 1, 0])

    else:
        res.extend([
            int(blending['enable']),
            BLEND_FUNC[blending.get('op_color', 'add')],
            BLEND_FUNC[blending.get('op_alpha', 'add')],
            BLEND_CONSTANT[blending.get('src_color', 'one')],
            BLEND_CONSTANT[blending.get('dst_color', 'zero')],
            BLEND_CONSTANT[blending.get('src_alpha', 'one')],
            BLEND_CONSTANT[blending.get('dst_alpha', 'zero')],
        ])

    if polygon_offset is False:
        res.extend([False, 0.0, 0.0])

    else:
        res.extend([
            True,
            float(polygon_offset['factor']),
            float(polygon_offset['units']),
        ])

    res.append(len(attachments[1]))
    return tuple(res)


def program(vertex_shader, fragment_shader, layout, includes):
    def include(match):
        name = match.group(1)
        content = includes.get(name)
        if content is None:
            raise KeyError(f'cannot include "{name}"')
        return content

    vert = textwrap.dedent(vertex_shader).strip()
    vert = re.sub(r'#include\s+"([^"]+)"', include, vert)
    vert = vert.encode().replace(b'\r', b'')

    frag = textwrap.dedent(fragment_shader).strip()
    frag = re.sub(r'#include\s+"([^"]+)"', include, frag)
    frag = frag.encode().replace(b'\r', b'')

    bindings = []
    for obj in sorted(layout, key=lambda x: x['name']):
        bindings.extend((obj['name'], obj['binding']))

    return (vert, 0x8b31), (frag, 0x8b30), tuple(bindings)


def compile_error(shader: bytes, shader_type: int, log: bytes):
    name = {0x8b31: 'Vertex Shader', 0x8b30: 'Fragment Shader'}[shader_type]
    log = log.rstrip(b'\x00').decode()
    raise ValueError(f'{name} Error\n\n{log}')


def linker_error(vertex_shader: bytes, fragment_shader: bytes, log: bytes):
    log = log.rstrip(b'\x00').decode()
    raise ValueError(f'Linker Error\n\n{log}')


def flatten(iterable):
    try:
        for x in iterable:
            yield from flatten(x)
    except TypeError:
        yield iterable


def clean_uniform_name(uniform):
    name = uniform['name']
    if name.endswith('[0]') and uniform['size'] > 1:
        return name[:-3]
    return name


def uniforms(uniforms, values):
    data = bytearray()
    uniform_map = {clean_uniform_name(obj): obj for obj in uniforms}

    if values == 'all':
        names = []
        for obj in uniforms:
            location = obj['location']
            size = obj['size']
            gltype = obj['type']
            if gltype not in UNIFORM_PACKER:
                continue
            names.append(clean_uniform_name(obj))
            items, format = UNIFORM_PACKER[gltype]
            data.extend(struct.pack('4i', size * items, location, size, gltype))
            data.extend(bytearray(size * items * 4))
        return names, data

    for name, value in values.items():
        if name not in uniform_map:
            raise KeyError(f'Uniform "{name}" does not exist')
        value = tuple(flatten(value))
        location = uniform_map[name]['location']
        size = uniform_map[name]['size']
        gltype = uniform_map[name]['type']
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

    return list(values), data


def validate(attributes, uniforms, uniform_buffers, vertex_buffers, layout, resources, limits):
    attributes = [
        {
            'name': obj['name'].replace('[0]', f'[{i:d}]'),
            'location': obj['location'] + i if obj['location'] >= 0 else -1,
        }
        for obj in attributes for i in range(obj['size'])
        if obj['name'] not in VERTEX_SHADER_BUILTINS
    ]
    uniforms = [
        {
            'name': obj['name'].replace('[0]', f'[{i}]'),
            'location': obj['location'] + i if obj['location'] >= 0 else -1,
        }
        for obj in uniforms for i in range(obj['size'])
        if obj['type'] not in UNIFORM_PACKER
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
    max_uniform_block_size = limits['max_uniform_block_size']

    for obj in uniform_buffers:
        if obj['size'] > max_uniform_block_size:
            raise ValueError(
                f'Uniform buffer "{obj["name"]}" is too large, the maximum supported size is {max_uniform_block_size}'
            )

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
        elif name in uniform_buffer_map:
            uniform_buffer_binding_map[binding] = obj
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
                raise ValueError(
                    f'Uniform buffer is too small {buffer.size} is less than '
                    f'{size} for "{name}" with binding {binding}'
                )
            bound_uniform_buffers.add(binding)
        elif resource_type == 'sampler':
            image = obj['image']
            if binding not in uniform_binding_map:
                raise ValueError(f'Sampler binding {binding} does not exist')
            name = uniform_binding_map[binding]['name']
            if binding in bound_uniforms:
                raise ValueError(f'Duplicate sampler binding for "{name}" with binding {binding}')
            if image.samples != 1:
                raise ValueError(f'Multisample images cannot be attached to "{name}" with binding {binding}')
            bound_uniforms.add(binding)
        else:
            raise ValueError(f'Invalid resource type "{resource_type}"')
