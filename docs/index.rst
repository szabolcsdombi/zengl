ZenGL
-----

ZenGL is a low level graphics library. Works on all platforms including the browser.

.. code::

    pip install zengl

- `Documentation <https://zengl.readthedocs.io/>`_
- `zengl on Github <https://github.com/szabolcsdombi/zengl/>`_
- `zengl on PyPI <https://pypi.org/project/zengl/>`_

Description
===========

- **Context** is the root object to access OpenGL
- **Image** is an OpenGL Texture or Renderbuffer
- **Buffer** is an OpenGL Buffer
- **Pipeline** is an OpenGL Program + Vertex Array + Framebuffer + *complete state for rendering*

.. code::

    ctx = zengl.context()
    texture = ctx.image(size, 'rgba8unorm', pixels)
    renderbuffer = ctx.image(size, 'rgba8unorm', samples=4)
    vertex_buffer = ctx.buffer(vertices)
    pipeline = ctx.pipeline(...)

The complete OpenGL state is encapsulated by the **Pipeline**.

Rendering with multiple pipelines guarantees proper state with minimal changes and api calls.

.. code::

    background.render()
    scene.render()
    particles.render()
    bloom.render()

**Pipelines** render to framebuffers, **Images** can be blit to the screen.

.. code::

    # init time
    pipeline = ctx.pipeline(
        framebuffer=[image, depth],
    )

.. code::

    # per frame
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()

Programs are simple, easy, and cached. Unique shader sources are only compiled once.

.. code::

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

Vertex Arrays are simple.

.. code::

    # simple
    pipeline = ctx.pipeline(
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
    )

.. code::

    # indexed
    pipeline = ctx.pipeline(
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        index_buffer=index_buffer,
        vertex_count=index_buffer.size // 4,
    )

.. code::

    # instanced
    pipeline = ctx.pipeline(
        vertex_buffers=[
            *zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
            *zengl.bind(instance_buffer, '3f 4f /i', 3, 4),
        ],
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
        instance_count=1000,
    )

Uniform Buffer, Texture, and Sampler binding is easy.

.. code::

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

.. code::

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

Postprocessing and Compute can be implemented as rendering a fullscreen quad.

.. code::

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

.. code::

    particle_system = ctx.pipeline(
        fragment_shader='''
            #version 330 core

            uniform sampler2D Position;
            uniform sampler2D Velocity;
            uniform vec3 Acceleration;

            layout (location = 0) out vec3 OutputPosition;
            layout (location = 1) out vec3 OutputVelocity;

            void main() {
                OutputPosition = Position + Velocity;
                OutputVelocity = Velocity + Acceleration;
            }
        ''',
    )

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

The `default framebuffer <https://www.khronos.org/opengl/wiki/Default_Framebuffer>`_ in OpenGL is highly dependent on how the Window is created.
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

ZenGL emerged from an experimental version of `ModernGL <https://github.com/moderngl/moderngl>`_.
To keep ModernGL backward compatible, ZenGL was re-designed from the ground-up to support a strict subset of OpenGL.
On the other hand, ModernGL supports a wide variety of OpenGL versions and extensions.

Disambiguation
==============

- ZenGL is a drop-in replacement for pure OpenGL code
- Using ZenGL requires some OpenGL knowledge
- ZenGL Images are OpenGL `Texture Objects <https://www.khronos.org/opengl/wiki/Texture>`_ or `Renderbuffer Objects <https://www.khronos.org/opengl/wiki/Renderbuffer_Object>`_
- ZenGL Buffers are OpenGL `Buffer Objects <https://www.khronos.org/opengl/wiki/Buffer_Object>`_
- ZenGL Pipelines contain an OpenGL `Vertex Array Object <https://www.khronos.org/opengl/wiki/Vertex_Specification#Vertex_Array_Object>`_, a `Program Object <https://www.khronos.org/opengl/wiki/GLSL_Object#Program_objects>`_, and a `Framebuffer Object <https://www.khronos.org/opengl/wiki/Framebuffer>`_
- ZenGL Pielines may also contain OpenGL `Sampler Objects <https://www.khronos.org/opengl/wiki/Sampler_Object>`_
- Creating ZenGL Pipelines does not necessarily compile the shader from source
- The ZenGL Shader Cache exists independently from the Pipeline objects
- A Framebuffer is always represented by a Python list of ZenGL Images
- There is no `Pipeline.clear()` method, individual images must be cleared independently
- GLSL Uniform Blocks and sampler2D objects are bound in the Pipeline layout
- Textures and Uniform Buffers are bound in the Pipeline resources

Documentation
=============

.. py:class:: Context

| Represents an OpenGL context.

.. py:class:: Buffer

| Represents an OpenGL buffer.

.. py:class:: Image

| Represents an OpenGL texture or renderbuffer.

.. py:class:: Pipeline

| Represents an entire rendering pipeline including the global state, shader program, framebuffer, vertex state,
  uniform buffer bindings, samplers, and sampler bindings.

Context
=======

.. py:method:: zengl.context() -> Context

All interactions with OpenGL are done by a Context object.
Only the first call to this method creates a new context.
Multiple calls return the previously created context.

.. py:method:: zengl.loader(headless: bool = False) -> ContextLoader

This method provides a default context loader.
Headless contexts require `glcontext` to be installed.

.. note::

    Implementing a context loader enables zengl to run in custom environments.
    ZenGL uses a subset of the OpenGL 3.3 core, the list of methods can be found in the project source.
    The implementation takes into account the OpenGL ES compatibility and can also work with a WebGL2 backend.

.. py:method:: zengl.init(loader: ContextLoader)

Initialize the OpenGL bindings.
This method is automatically called by :py:meth:`zengl.context`.

**Context for a window**

.. code-block::

    ctx = zengl.context()

**Context for headless rendering**

.. code-block::

    zengl.init(zengl.loader(headless=True))
    ctx = zengl.context()

**Rendering**

.. py:method:: Context.new_frame(reset: bool = True, clear: bool = True, frame_time: bool = False)

**reset**
    | A boolean to clear ZenGL internals assuming OpenGL global state.

**clear**
    | A boolean to clear the default framebuffer.

**frame_time**
    | A boolean to start a query with ``GL_TIME_ELAPSED``.
    | The :py:attr:`Context.frame_time` is set by :py:meth:`Context.end_frame`.

.. py:method:: Context.end_frame(clean: bool = True, flush: bool = True, sync: bool = False)

**clean**
    | A boolean to unset OpenGL object bindings managed by ZenGL.
    | The values are not restored from any previous states, they are set to zero.

**flush**
    | A boolean to call ``glFlush``.

**sync**
    | A boolean to wait for a ``glFenceSync``.

Buffer
======

| Buffers hold vertex, index, and uniform data used by rendering.
| Buffers have a fixed size allocated upfront in the device memory.

.. code-block::

    vertex_buffer = ctx.buffer(open('sphere.mesh', 'rb').read())

.. code-block::

    vertex_buffer = ctx.buffer(np.array([0.0, 0.0, 1.0, 1.0], 'f4'))

.. code-block::

    index_buffer = ctx.buffer(np.array([0, 1, 2], 'i4'), index=True)

.. code-block::

    vertex_buffer = ctx.buffer(size=1024)

.. py:method:: Context.buffer(data, size, access, index, uniform, external) -> Buffer

**data**
    | The buffer content, represented as ``bytes`` or a buffer for example a numpy array.
    | If the data is None the content of the buffer will be uninitialized and the size is mandatory.
    | The default value is None.

**size**
    | The size of the buffer. It must be None if the data parameter was provided.
    | The default value is None and it means the size of the data.

**access**
    | Specifies the expected access pattern of the data store.
    | Possible values are:
    | - "stream_draw"
    | - "stream_read"
    | - "stream_copy"
    | - "static_draw"
    | - "static_read"
    | - "static_copy"
    | - "dynamic_draw"
    | - "dynamic_read"
    | - "dynamic_copy"

**index**
    | Modifies the write operation to use the element array buffer binding.
    | The default value is False.

**uniform**
    | Modifies the write operation to use the uniform buffer binding.
    | The default value is False.

**external**
    | An OpenGL Buffer Object returned by glGenBuffers.
    | The default value is 0.

.. py:method:: Buffer.write(data, offset)

**data**
    | The content to be written into the buffer, represented as ``bytes`` or a buffer.

**offset**
    | An int, representing the write offset in bytes.

.. py:method:: Buffer.read(size, offset, into) -> bytes

**size**
    | An int, representing the size of the buffer in bytes to be read.
    | The default value is None and it means the entire buffer.

**offset**
    | An int, representing the offset in bytes for the read.
    | When the offset is not None the size must also be defined.
    | The default value is None and it means the beginning of the buffer.

.. py:method:: Buffer.view(size, offset) -> BufferView

.. py:attribute:: Buffer.size

    An int, representing the size of the buffer in bytes.

Image
=====

| Images hold texture data or render outputs.
| Images with texture support are implemented with OpenGL textures.
| Render outputs that are not sampled from the shaders are using renderbuffers instead.

**render targets**

.. code-block::

    image = ctx.image(window.size, 'rgba8unorm', samples=4)
    depth = ctx.image(window.size, 'depth24plus', samples=4)
    framebuffer = [image, depth]

**textures**

.. code-block::

    img = Image.open('example.png').convert('RGBA')
    texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())

.. py:method:: Context.image(size, format, data, samples, array, texture, cubemap, external) -> Image

**size**
    | The image size as a tuple of two ints.

**format**
    | The image format represented as string. (:ref:`list of image format<Image Formats>`)
    | The two most common are ``'rgba8unorm'`` and ``'depth24plus'``

**data**
    | The image content, represented as ``bytes`` or a buffer for example a numpy array.
    | If the data is None the content of the image will be uninitialized. The default value is None.

**samples**
    | The number of samples for the image. Multisample render targets must have samples > 1.
    | Textures must have samples = 1. Only powers of two are possible. The default value is 1.
    | For multisampled rendering usually 4 is a good choice.

**array**
    | The number of array layers for the image. For non-array textures, the value must be 0.
    | The default value is 0.

**texture**
    | A boolean representing the image to be sampled from shaders or not.
    | For textures, this flag must be True, for render targets it should be False.
    | Multisampled textures to be sampled from the shaders are not supported.
    | The default is None and it means to be determined from the image type.

**cubemap**
    | A boolean representing the image to be a cubemap texture. The default value is False.

**external**
    | An OpenGL Texture Object returned by glGenTextures.
    | The default value is 0.

.. py:method:: Image.blit(target, offset, size, crop, filter)

**target**
    | The target image to copy to. The default value is None and it means to copy to the screen.

**size and offset**
    | The size and offset, defining a sub-part of the image to copy to.

**size**
    | The size of the target image area to copy to. The default value is None and it means the full size of the image.

**crop**
    | The crop area in the source image to copy from. The default value is None and it means the full size of the image.

**filter**
    | A boolean to enable linear filtering for scaled images. By default it is True.
      It has no effect if the source and target viewports have the same size.

.. py:method:: Image.clear()

Clear the image with the :py:attr:`Image.clear_value`

.. py:method:: Image.mipmaps(base, levels)

Generate mipmaps for the image.

**base**
    | The base image level. The default value is 0.

**levels**
    | The number of mipmap levels to generate starting from the base.
    | The default is None and it means to generate mipmaps all the mipmap levels.

.. py:method:: Image.read(size, offset) -> bytes

**size and offset**
    | The size and offset, defining a sub-part of the image to be read.
    | Both the size and offset are tuples of two ints.
    | The size is mandatory when the offset is not None.
    | By default the size is None and it means the full size of the image.
    | By default the offset is None and it means a zero offset.

.. py:method:: Image.write(data, size, offset, layer, level) -> bytes

**data**
    | The content to be written to the image represented as ``bytes`` or a buffer for example a numpy array.

**size and offset**
    | The size and offset, defining a sub-part of the image to be read.
    | Both the size and offset are tuples of two ints.
    | The size is mandatory when the offset is not None.
    | By default the size is None and it means the full size of the image.
    | By default the offset is None and it means a zero offset.

**layer**
    | An int representing the layer to be written to.
    | This value must be None for non-layered textures.
    | For array and cubemap textures, the layer must be specified.
    | The default value is None and it mean all the layers.

**level**
    | An int representing the mipmap level to be written to.
    | The default value is 0.

.. py:attribute:: Image.clear_value

| The clear value for the image used by the :py:meth:`Image.clear`
| For the color and stencil components, the default value is zero. For depth, the default value is 1.0
| For single component images, the value is float or int depending on the image type.
| For multi-component images, the value is a tuple of ints or floats.
| The clear value type for the ``depth24plus-stencil8`` format is a tuple of float and int.

.. py:attribute:: Image.size

| The image size as a tuple of two ints.

.. py:attribute:: Image.samples

| The number of samples the image has.

.. py:attribute:: Image.color

| A boolean representing if the image is a color image.
| For depth and stencil images this value is False.

Pipeline
========

.. py:method:: Context.pipeline(vertex_shader, fragment_shader, layout, resources, uniforms, depth, stencil, blend, framebuffer, vertex_buffers, index_buffer, short_index, cull_face, topology, vertex_count, instance_count, first_vertex, viewport, uniform_data, viewport_data, render_data, includes, template) -> Pipeline

**vertex_shader**
    | The vertex shader code.

**fragment_shader**
    | The fragment shader code.

**layout**
    | Layout binding definition for the uniform buffers and samplers.

**resources**
    | The list of uniform buffers and samplers to be bound.

**uniforms**
    | The default values for uniforms.

**depth**
    | The depth settings

**stencil**
    | The stencil settings

**blend**
    | The blend settings

**framebuffer**
    | A list of images representing the framebuffer for the rendering.
    | The depth or stencil attachment must be the last one in the list.
    | The size and number of samples of the images must match.

**vertex_buffers**
    | A list of vertex attribute bindings with the following keys:

        | **buffer:** A buffer to be used as the vertex attribute source
        | **format:** The vertex attribute format. (:ref:`list of vertex formats<Vertex Formats>`)
        | **location:** The vertex attribute location
        | **offset:** The buffer offset in bytes
        | **stride:** The stride in bytes
        | **step:** ``'vertex'`` for per-vertex attributes. ``'instance'`` for per-instance attributes

    The :py:meth:`zengl.bind` method produces this list in a more compact form.

**index_buffer**
    | A buffer object to be used as the index buffer.
    | The default value is None and it means to disable indexed rendering.

**short_index**
    | A boolean to enable ``GL_UNSIGNED_SHORT`` as the index type.
    | When this flag is False the ``GL_UNSIGNED_INT`` is used.
    | The default value is False.

**cull_face**
    | A string representing the cull face. It must be ``'front'``, ``'back'`` or ``'none'``
    | The default value is ``'none'``

**topology**
    | A string representing the rendered primitive topology.
    | It must be one of the following:

        - ``'points'``
        - ``'lines'``
        - ``'line_loop'``
        - ``'line_strip'``
        - ``'triangles'``
        - ``'triangle_strip'``
        - ``'triangle_fan'``

    | The default value is ``'triangles'``

**vertex_count**
    | The number of vertices or the number of elements to draw.

**instance_count**
    | The number of instances to draw.

**first_vertex**
    | The first vertex or the first index to start drawing from.
    | The default value is 0. This is a mutable parameter at runtime.

**viewport**
    | The render viewport, defined as tuples of four ints in (x, y, width, height) format.
    | The default is the full size of the framebuffer.

**uniform_data**
    | Memoryview to use as the source of uniform values.

**viewport_data**
    | Memoryview to use as the source of viewport value.
    | It must points to a memory of (x, y, width, height) integers.

**render_data**
    | Memoryview to use as the source of render parameters.
    | It must points to a memory of (vertex_count, instance_count, first_vertex) integers.

**includes**
    | A dictionary to use for resolving the includes.
    | The default value is None and it means :py:attr:`Context.includes`.

**template**
    | A Pipeline object to use as the default settings.
    | Setting a template fixes the shader source and layout definition.

.. py:attribute:: Pipeline.vertex_count

    | The number of vertices or the number of elements to draw.

.. py:attribute:: Pipeline.instance_count

    | The number of instances to draw.

.. py:attribute:: Pipeline.first_vertex

    | The first vertex or the first index to start drawing from.

.. py:attribute:: Pipeline.viewport

    | The render viewport, defined as tuples of four ints in (x, y, width, height) format.

.. py:attribute:: Pipeline.uniforms

    | The uniform values as memoryviews.

.. py:method:: Pipeline.render()

    | Execute the rendering pipeline.

Shader Code
===========

- **do** use ``#version 330 core`` or ``#version 300 es`` as the first line in the shader.
- **do** use ``layout (std140)`` for uniform buffers.
- **do** use ``layout (location = ...)`` for the vertex shader inputs.
- **do** use ``layout (location = ...)`` for the fragment shader outputs.

- **don't** use ``layout (location = ...)`` for the vertex shader outputs or the fragment shader inputs.
  Matching name and order are sufficient and much more readable.

- **don't** use ``layout (binding = ...)`` for the uniform buffers or samplers.
  It is not a core feature in OpenGL 3.3 and ZenGL enforces the program layout from the pipeline parameters.

- **do** use uniform buffers, use a single one if possible.
- **don't** use uniforms, use uniform buffers instead.
- **don't** put constants in uniform buffers, use ``#include`` and string formatting.
- **don't** over-use the ``#include`` statement.
- **do** use includes without extensions.

- **do** arrange pipelines in such an order to minimize framebuffer then program changes.

Shader Includes
===============

| Shader includes were designed to solve a single problem of sharing code among shaders without having to field format the shader code.
| Includes are simple string replacements from :py:attr:`Context.includes`
| The include statement stands for including constants, functions, logic or behavior, but not files. Hence the naming should not contain extensions like ``.h``
| Nested includes do not work, they are overcomplicated and could cause other sorts of issues.

.. py:attribute:: Context.includes

    | A string to string mapping dict.

**Example**

.. code-block::

    ctx.includes['common'] = '...'

    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            #include "common"
            #include "qtransform"

            void main() {
            }
        ''',
    )

Include Patterns
================

**common uniform buffer**

.. code-block::

    ctx.includes['common'] = '''
        layout (std140) uniform Common {
            mat4 mvp;
        };
    '''

**quaternion transform**

.. code-block::

    ctx.includes['qtransform'] = '''
        vec3 qtransform(vec4 q, vec3 v) {
            return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
        }
    '''

**gaussian filter**

.. code-block::

    def kernel(s):
        x = np.arange(-s, s + 1)
        y = np.exp(-x * x / (s * s / 4))
        y /= y.sum()
        v = ', '.join(f'{t:.8f}' for t in y)
        return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[ <{v});'

    ctx.includes['kernel'] = kernel(19)

**hsv to rgb**

.. code-block::

    ctx.includes['hsv2rgb'] = '''
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
    '''

Rendering to Texture
====================

Rendering to texture is supported. However, multisampled images must be downsampled before being used as textures.
In that case, an intermediate render target must be samples > 1 and texture = False.
Then this image can be downsampled with :py:meth:`Image.blit` to another image with samples = 1 and texture = True.

Cleanup
=======

Clean only if necessary. It is ok not to clean up before the program ends.

.. py:method:: Context.release(obj: Buffer | Image | Pipeline | str)

This method releases the OpenGL resources associated with the parameter.
OpenGL resources are not released automatically on garbage collection.
Release Pipelines before the Images and Buffers they use.

When the string ``shader_cache`` is passed to this method,
it calls glDeleteShader for all the previously created vertex and fragment shader modules.

When the string ``all`` is passed to this method, it releases all the resources allocated from this context.

Interoperability
================

| Some window implementations expose a framebuffer object for drawing.
| Detecting this framebuffer is an error-prone and non-reliable solution.
| The recommended way is to change :py:attr:`Context.screen` to the framebuffer object.

| Running zengl alongside other rendering libraries is not recommended.
| However, to port existing code to zengl, some interoperability may be necessary.
| OpenGL objects can be extracted with :py:meth:`zengl.inspect`.
| It is possible to interact with these objects using the OpenGL API directly.

.. py:method:: zengl.inspect(obj: Buffer | Image | Pipeline)

Returns a dictionary with all of the OpenGL objects.

.. py:attribute:: Context.screen

| An integer representing the default framebuffer object.
| You may want to change this attribute when using PyQt.

Utils
=====

.. py:attribute:: Context.info

- vendor
- renderer
- version
- glsl
- max_uniform_buffer_bindings
- max_uniform_block_size
- max_combined_uniform_blocks
- max_combined_texture_image_units
- max_vertex_attribs
- max_draw_buffers
- max_samples

.. py:attribute:: Context.frame_time

| An int representing the time elapsed between the :py:meth:`Context.new_frame` and :py:meth:`Context.end_frame`.
| The value is in nanoseconds and it is zero if the frame_time was not enabled.

.. py:method:: zengl.camera(eye, target, up, fov, aspect, near, far, size, clip) -> bytes

| Returns a Model-View-Projection matrix for uniform buffers.
| The return value is bytes and can be used as a parameter for :py:meth:`Buffer.write`.

.. code-block::

    mvp = zengl.camera(eye=(4.0, 3.0, 2.0), target=(0.0, 0.0, 0.0), aspect=16.0 / 9.0, fov=45.0)

.. py:method:: zengl.bind(buffer: Buffer, layout: str, *attributes: int) -> List[VertexBufferBinding]

| Helper function for binding a single buffer to multiple vertex attributes.
| The -1 is a special value allowed in the attributes to represent not yet implemented attributes.
| An ending ``/i`` is allowed in the layout to represent per instance stepping.

.. py:method:: zengl.calcsize(layout: str) -> int

| Calculates the size of a vertex attribute buffer layout.

.. _Image Formats:

Image Formats
=============

==================== ===================== ================== =================
ZenGL format         internal format       format             type
==================== ===================== ================== =================
r8unorm              GL_R8                 GL_RED             GL_UNSIGNED_BYTE
rg8unorm             GL_RG8                GL_RG              GL_UNSIGNED_BYTE
rgba8unorm           GL_RGBA8              GL_RGBA            GL_UNSIGNED_BYTE
r8snorm              GL_R8_SNORM           GL_RED             GL_UNSIGNED_BYTE
rg8snorm             GL_RG8_SNORM          GL_RG              GL_UNSIGNED_BYTE
rgba8snorm           GL_RGBA8_SNORM        GL_RGBA            GL_UNSIGNED_BYTE
r8uint               GL_R8UI               GL_RED_INTEGER     GL_UNSIGNED_BYTE
rg8uint              GL_RG8UI              GL_RG_INTEGER      GL_UNSIGNED_BYTE
rgba8uint            GL_RGBA8UI            GL_RGBA_INTEGER    GL_UNSIGNED_BYTE
r16uint              GL_R16UI              GL_RED_INTEGER     GL_UNSIGNED_SHORT
rg16uint             GL_RG16UI             GL_RG_INTEGER      GL_UNSIGNED_SHORT
rgba16uint           GL_RGBA16UI           GL_RGBA_INTEGER    GL_UNSIGNED_SHORT
r32uint              GL_R32UI              GL_RED_INTEGER     GL_UNSIGNED_INT
rg32uint             GL_RG32UI             GL_RG_INTEGER      GL_UNSIGNED_INT
rgba32uint           GL_RGBA32UI           GL_RGBA_INTEGER    GL_UNSIGNED_INT
r8sint               GL_R8I                GL_RED_INTEGER     GL_BYTE
rg8sint              GL_RG8I               GL_RG_INTEGER      GL_BYTE
rgba8sint            GL_RGBA8I             GL_RGBA_INTEGER    GL_BYTE
r16sint              GL_R16I               GL_RED_INTEGER     GL_SHORT
rg16sint             GL_RG16I              GL_RG_INTEGER      GL_SHORT
rgba16sint           GL_RGBA16I            GL_RGBA_INTEGER    GL_SHORT
r32sint              GL_R32I               GL_RED_INTEGER     GL_INT
rg32sint             GL_RG32I              GL_RG_INTEGER      GL_INT
rgba32sint           GL_RGBA32I            GL_RGBA_INTEGER    GL_INT
r16float             GL_R16F               GL_RED             GL_FLOAT
rg16float            GL_RG16F              GL_RG              GL_FLOAT
rgba16float          GL_RGBA16F            GL_RGBA            GL_FLOAT
r32float             GL_R32F               GL_RED             GL_FLOAT
rg32float            GL_RG32F              GL_RG              GL_FLOAT
rgba32float          GL_RGBA32F            GL_RGBA            GL_FLOAT
depth16unorm         GL_DEPTH_COMPONENT16  GL_DEPTH_COMPONENT GL_UNSIGNED_SHORT
depth24plus          GL_DEPTH_COMPONENT24  GL_DEPTH_COMPONENT GL_UNSIGNED_INT
depth24plus-stencil8 GL_DEPTH_COMPONENT24  GL_DEPTH_COMPONENT GL_UNSIGNED_INT
depth32float         GL_DEPTH_COMPONENT32F GL_DEPTH_COMPONENT GL_FLOAT
==================== ===================== ================== =================

.. _Vertex Formats:

Vertex Formats
==============

========== ============= ================== ==== ==========
ZenGL bind ZenGL format  type               size normalized
========== ============= ================== ==== ==========
1f         float32       GL_FLOAT           1    no
2f         float32x2     GL_FLOAT           2    no
3f         float32x3     GL_FLOAT           3    no
4f         float32x4     GL_FLOAT           4    no
1u         uint32        GL_UNSIGNED_INT    1    no
2u         uint32x2      GL_UNSIGNED_INT    2    no
3u         uint32x3      GL_UNSIGNED_INT    3    no
4u         uint32x4      GL_UNSIGNED_INT    4    no
1i         sint32        GL_INT             1    no
2i         sint32x2      GL_INT             2    no
3i         sint32x3      GL_INT             3    no
4i         sint32x4      GL_INT             4    no
2u1        uint8x2       GL_UNSIGNED_BYTE   2    no
4u1        uint8x4       GL_UNSIGNED_BYTE   4    no
2i1        sint8x2       GL_BYTE            2    no
4i1        sint8x4       GL_BYTE            4    no
2h         float16x2     GL_HALF_FLOAT      2    no
4h         float16x4     GL_HALF_FLOAT      4    no
2nu1       unorm8x2      GL_UNSIGNED_BYTE   2    yes
4nu1       unorm8x4      GL_UNSIGNED_BYTE   4    yes
2ni1       snorm8x2      GL_BYTE            2    yes
4ni1       snorm8x4      GL_BYTE            4    yes
2u2        uint16x2      GL_UNSIGNED_SHORT  2    no
4u2        uint16x4      GL_UNSIGNED_SHORT  4    no
2i2        sint16x2      GL_SHORT           2    no
4i2        sint16x4      GL_SHORT           4    no
2nu2       unorm16x2     GL_UNSIGNED_SHORT  2    yes
4nu2       unorm16x4     GL_UNSIGNED_SHORT  4    yes
2ni2       snorm16x2     GL_SHORT           2    yes
4ni2       snorm16x4     GL_SHORT           4    yes
========== ============= ================== ==== ==========
