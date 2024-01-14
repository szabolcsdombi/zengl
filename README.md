# ZenGL

[![ZenGL](https://repository-images.githubusercontent.com/420309094/f7c17e13-4d5b-4a38-8b52-ab2dfdacd5a0)](#zengl)

```
pip install zengl
```

- [Documentation](https://zengl.readthedocs.io/)
- [zengl on Github](https://github.com/szabolcsdombi/zengl/)
- [zengl on PyPI](https://pypi.org/project/zengl/)

## Concept

ZenGL provides a simple, structured way to render with OpenGL in Python.

Pipelines are self-contained, no global state affects the render.

State changes between pipelines are optimized; framebuffers, descriptor sets are re-used.

ZenGL is a low level library, it adds no magic on the rendering side. All you need to know is OpenGL.

ZenGL runs Natively (Desktop OpenGL), on top of Angle (DirectX, Vulkan, Metal), or WebGL2 (In the Browser).

## [Examples](./examples/)

[![bezier_curves](https://user-images.githubusercontent.com/11232402/235417415-f04815bf-3380-45fa-9804-f9f36016f46c.png)](#native-examples)
[![deferred_rendering](https://user-images.githubusercontent.com/11232402/235417431-4dd870ea-1804-4b00-bfd2-49e3ca72e2b1.png)](#native-examples)
[![envmap](https://user-images.githubusercontent.com/11232402/235417438-0cc02333-dd92-47e4-b874-ff1b6dca2086.png)](#native-examples)
[![fractal](https://user-images.githubusercontent.com/11232402/235417445-73efbe67-21ea-4aae-a1ff-6aa4002bf58d.png)](#native-examples)
[![grass](https://user-images.githubusercontent.com/11232402/235417450-3ff0b82d-e097-40cd-947a-58803e464cd3.png)](#native-examples)
[![normal_mapping](https://user-images.githubusercontent.com/11232402/235417454-1d8e4bfb-02ad-42a2-87ba-ce39f47de14d.png)](#native-examples)
[![rigged_objects](https://user-images.githubusercontent.com/11232402/235417459-79483b7f-6581-4788-a662-ef81087334b6.png)](#native-examples)
[![wireframe](https://user-images.githubusercontent.com/11232402/235417465-f3f54a9b-624b-4fa1-88b6-f725ac468e78.png)](#native-examples)

### Complete Pipeline Definition

Probably the only documentation needed.

```py
pipeline = ctx.pipeline(
    # program definition
    vertex_shader='...',
    fragment_shader='...',
    layout=[
        {
            'name': 'Uniforms',
            'binding': 0,
        },
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],

    # descriptor sets
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

    # uniforms
    uniforms={
        'color': [0.0, 0.5, 1.0],
        'iterations': 10,
    },

    # program definition global state
    depth={
        'func': 'less',
        'write': False,
    },
    stencil={
        'front': {
            'fail_op': 'replace',
            'pass_op': 'replace',
            'depth_fail_op': 'replace',
            'compare_op': 'always',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
        'back': ...,
        # or
        'both': ...,
    },
    blend={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha',
    },
    cull_face='back',
    topology='triangles',

    # framebuffer
    framebuffer=[color1, color2, ..., depth],
    viewport=(x, y, width, height),

    # vertex array
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f', 0, 1), # bound vertex attributes
        *zengl.bind(None, '2f', 2), # unused vertex attribute
    ],
    index_buffer=index_buffer, # or None
    short_index=False, # 2 or 4 byte intex
    vertex_count=...,
    instance_count=1,
    first_vertex=0,

    # override includes
    includes={
        'common': '...',
    },
)

# some members are actually mutable and calls no OpenGL functions
pipeline.viewport = ...
pipeline.vertex_count = ...
pipeline.uniforms['iterations'][:] = struct.pack('i', 50) # writable memoryview

# rendering
pipeline.render() # no parameters for hot code
```
