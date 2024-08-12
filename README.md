[![ZenGL](https://repository-images.githubusercontent.com/420309094/f7c17e13-4d5b-4a38-8b52-ab2dfdacd5a0)](#zengl)

```
pip install zengl
```

- [Documentation](https://zengl.readthedocs.io/)
- [zengl on Github](https://github.com/szabolcsdombi/zengl/)
- [zengl on PyPI](https://pypi.org/project/zengl/)
- [Discord](https://discord.gg/nM34Uv7x)

# ZenGL

ZenGL is a low level graphics library. Works on all platforms including the browser.

## Description

- **Context** is the root object to access OpenGL
- **Image** is an OpenGL Texture or Renderbuffer
- **Buffer** is an OpenGL Buffer
- **Pipeline** is an OpenGL Program + Vertex Array + Framebuffer + _complete state for rendering_

```py
ctx = zengl.context()
texture = ctx.image(size, 'rgba8unorm', pixels)
renderbuffer = ctx.image(size, 'rgba8unorm', samples=4)
vertex_buffer = ctx.buffer(vertices)
pipeline = ctx.pipeline(...)
```

The complete OpenGL state is encapsulated by the **Pipeline**.

Rendering with multiple pipelines guarantees proper state with minimal changes and api calls.

```py
background.render()
scene.render()
particles.render()
bloom.render()
```

**Pipelines** render to framebuffers, **Images** can be blit to the screen.

```py
# init time
pipeline = ctx.pipeline(
    framebuffer=[image, depth],
)
```

```py
# per frame
image.clear()
depth.clear()
pipeline.render()
image.blit()
```

Programs are simple, easy, and cached. Unique shader sources are only compiled once.

```py
pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        void main() {
            gl_Position = ...
        }
    ''',
    fragment_shader='''
        #version 330 core

        out vec4 frag_color;

        void main() {
            frag_color = ...
        }
    ''',
)
```

Vertex Arrays are simple.

```py
# simple
pipeline = ctx.pipeline(
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)
```

```py
# indexed
pipeline = ctx.pipeline(
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
)
```

```py
# instanced
pipeline = ctx.pipeline(
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        *zengl.bind(instance_buffer, '3f 4f /i', 3, 4),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
    instance_count=1000,
)
```

Uniform Buffer, Texture, and Sampler binding is easy.

```py
# uniform buffers
pipeline = ctx.pipeline(
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
)
```

```py
# textures
pipeline = ctx.pipeline(
    layout=[
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
)
```

Postprocessing and Compute can be implemented as rendering a fullscreen quad.

```py
pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec2 vertices[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        out vec4 frag_color;

        void main() {
            frag_color = ...
        }
    ''',
    topology='triangles',
    vertex_count=3,
)
```

```py
particle_system = ctx.pipeline(
    vertex_shader=...,
    fragment_shader='''
        #version 330 core

        uniform sampler2D Position;
        uniform sampler2D Velocity;
        uniform vec3 Acceleration;

        layout (location = 0) out vec3 OutputPosition;
        layout (location = 1) out vec3 OutputVelocity;

        void main() {
            ivec2 at = ivec2(gl_FragCoord.xy);
            vec3 position = texelFetch(Position, at, 0).xyz;
            vec3 velocity = texelFetch(Velocity, at, 0).xyz;
            OutputPosition = position + velocity;
            OutputVelocity = velocity + Acceleration;
        }
    ''',
)
```

ZenGL intentionally does not support:

- Transform Feedback
- Geometry Shaders
- Tesselation
- Compute Shaders
- 3D Textures
- Storage Buffers

Most of the above can be implemented in a more hardware friendly way using the existing ZenGL API.
Interoperability with other modules is also possible. Using such may reduce the application's portablity.
It is even possible to use direct OpenGL calls together with ZenGL, however this is likely not necessary.

It is common to render directly to the screen with OpenGL.
With ZenGL, the right way is to render to a framebuffer and blit the final image to the screen.
This allows fine-grained control of the framebuffer format, guaranteed multisampling settings, correct depth/stencil precison.
It is also possible to render directly to the screen, however this feature is designed to be used for the postprocessing step.

This design allows ZenGL to support:

- Rendering without a window
- Rendering to multiple windows
- Rendering to HDR monitors
- Refreshing the screen without re-rendering the scene
- Apply post-processing without changing how the scene is rendered
- Making reusable shaders and components
- Taking screenshots or exporting a video

The [default framebuffer](https://www.khronos.org/opengl/wiki/Default_Framebuffer) in OpenGL is highly dependent on how the Window is created.
It is often necessary to configure the Window to provide the proper depth precision, stencil buffer, multisampling and double buffering.
Often the "best pixel format" lacks all of these features on purpose. ZenGL aims to allow choosing these pixel formats and ensures the user specifies the rendering requirements.
It is even possible to render low-resolution images and upscale them for high-resolution monitors.
Tearing can be easily prevented by decoupling the scene rendering from the screen updates.

ZenGL was designed for Prototyping

It is tempting to start a project with Vulkan, however even getting a simple scene rendered requires tremendous work and advanced tooling to compile shaders ahead of time. ZenGL provides self-contained Pipelines which can be easily ported to Vulkan.
ZenGL code is verbose and easy to read.

ZenGL support multiple design patters

Many libraries enfore certain design patterns.
ZenGL avoids this by providing cached pipeline creation, pipeline templating and lean resourece and framebuffer definition.
It is supported to create pipelines on the fly or template them for certain use-cases.

> TODO: examples for such patters

ZenGL emerged from an experimental version of [ModernGL](https://github.com/moderngl/moderngl).
To keep ModernGL backward compatible, ZenGL was re-designed from the ground-up to support a strict subset of OpenGL.
On the other hand, ModernGL supports a wide variety of OpenGL versions and extensions.

## Disambiguation

- ZenGL is a drop-in replacement for pure OpenGL code
- Using ZenGL requires some OpenGL knowledge
- ZenGL Images are OpenGL [Texture Objects](https://www.khronos.org/opengl/wiki/Texture) or [Renderbuffer Objects](https://www.khronos.org/opengl/wiki/Renderbuffer_Object)
- ZenGL Buffers are OpenGL [Buffer Objects](https://www.khronos.org/opengl/wiki/Buffer_Object)
- ZenGL Pipelines contain an OpenGL [Vertex Array Object](https://www.khronos.org/opengl/wiki/Vertex_Specification#Vertex_Array_Object), a [Program Object](https://www.khronos.org/opengl/wiki/GLSL_Object#Program_objects), and a [Framebuffer Object](https://www.khronos.org/opengl/wiki/Framebuffer)
- ZenGL Pipelines may also contain OpenGL [Sampler Objects](https://www.khronos.org/opengl/wiki/Sampler_Object)
- Creating ZenGL Pipelines does not necessarily compile the shader from source
- The ZenGL Shader Cache exists independently from the Pipeline objects
- A Framebuffer is always represented by a Python list of ZenGL Images
- There is no `Pipeline.clear()` method, individual images must be cleared independently
- GLSL Uniform Blocks and sampler2D objects are bound in the Pipeline layout
- Textures and Uniform Buffers are bound in the Pipeline resources

## [Examples](./examples/)

[![bezier_curves](https://user-images.githubusercontent.com/11232402/235417415-f04815bf-3380-45fa-9804-f9f36016f46c.png)](#native-examples)
[![deferred_rendering](https://user-images.githubusercontent.com/11232402/235417431-4dd870ea-1804-4b00-bfd2-49e3ca72e2b1.png)](#native-examples)
[![envmap](https://user-images.githubusercontent.com/11232402/235417438-0cc02333-dd92-47e4-b874-ff1b6dca2086.png)](#native-examples)
[![fractal](https://user-images.githubusercontent.com/11232402/235417445-73efbe67-21ea-4aae-a1ff-6aa4002bf58d.png)](#native-examples)
[![grass](https://user-images.githubusercontent.com/11232402/235417450-3ff0b82d-e097-40cd-947a-58803e464cd3.png)](#native-examples)
[![normal_mapping](https://user-images.githubusercontent.com/11232402/235417454-1d8e4bfb-02ad-42a2-87ba-ce39f47de14d.png)](#native-examples)
[![rigged_objects](https://user-images.githubusercontent.com/11232402/235417459-79483b7f-6581-4788-a662-ef81087334b6.png)](#native-examples)
[![wireframe](https://user-images.githubusercontent.com/11232402/235417465-f3f54a9b-624b-4fa1-88b6-f725ac468e78.png)](#native-examples)

### Simple Pipeline Definition

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

