ZenGL
-----

ZenGL is a minimalist Python module providing exactly **one** way to render scenes with OpenGL.

.. code::

    pip install zengl

- `Documentation <https://zengl.readthedocs.io/>`_
- `zengl on Github <https://github.com/szabolcsdombi/zengl/>`_
- `zengl on PyPI <https://pypi.org/project/zengl/>`_

**ZenGL is ...**

- **high-performance**
- **simple** - *buffers, images, pipelines and there you go*
- **easy-to-learn** - *it is simply OpenGL with no magic added*
- **verbose** - *most common mistakes are catched and reported in a clear and understandable way*
- **robust** - *there is no global state or external trouble-maker affecting the render*
- **backward-compatible** - *it requires OpenGL 3.3 - it is just enough*
- **cached** - *most OpenGL objects are reused between renders*
- **zen** - *there is one way to do it*

.. warning::

    This document is still in draft

.. py:class:: Context

| Represents an OpenGL context.

.. py:class:: Buffer

| Represents an OpenGL buffer.

.. py:class:: Image

| Represents an OpenGL texture or renderbuffer.

.. py:class:: Pipeline

| Represents an entire rendering pipeline including the global state, shader program, framebuffer, vertex state,
  uniform buffer bindings, samplers and sampler bindings.

Context
-------

.. py:method:: zengl.context(loader: ContextLoader) -> Context

All interactions with OpenGL is done by a Context object.
There should be a single Context created per application.
A Context is created with the help of a context loader.
A context loader is an object implementing the load method to resolve OpenGL functions by name.
This enables zengl to be entirely platform independent.

.. py:method:: zengl.loader(headless: bool = False) -> ContextLoader

This method provides a default context loader. It requires `glcontext` to be installed.
ZenGL does not implement OpenGL function loading. glcontext is used when no alternatives are provided.

.. note::

    Implementing a context loader enables zengl to run in custom environments.
    ZenGL uses a subset of the OpenGL 3.3 core, the list of methods can be found in the project source.

**Context for a window**

.. code-block::

    ctx = zengl.context()

**Context for headless rendering**

.. code-block::

    ctx = zengl.context(zengl.loader(headless=True))

Buffer
------

| Buffers hold vertex, index and uniform data used by rendering.
| Buffers are not variable sized, they are allocated upfront in the device memory.

.. code-block::

    vertex_buffer = ctx.buffer(np.array([0.0, 0.0, 1.0, 1.0], 'f4'))

.. code-block::

    index_buffer = ctx.buffer(np.array([0, 1, 2], 'i4'))

.. code-block::

    vertex_buffer = ctx.buffer(size=1024)

.. py:method:: Context.buffer(data, size, dynamic) -> Buffer

**data**
    | The buffer content represented as ``bytes`` or a buffer for example a numpy array.
    | If the data is None the content of the buffer will be uninitialized and the size is mandatory.
    | The default value is None.

**size**
    | The size of the buffer. It must be None if the data parameter was provided.
    | The default value is None and it means the size of the data.

**dynamic**
    | A boolean to enable ``GL_DYNAMIC_DRAW`` on buffer creation.
    | When this flag is False the ``GL_STATIC_DRAW`` is used.
    | The default value is True.

.. py:method:: Buffer.write(data, offset)

**data**
    | The content to be written into the buffer represented as ``bytes`` or a buffer.

**offset**
    | An int representing the write offset in bytes.

.. py:method:: Buffer.map(size, offset, discard) -> memoryview

**size**
    | An int representing the size of the buffer in bytes to be mapped.
    | The default value is None and it means the entire buffer.

**offset**
    | An int representing the offset in bytes for the mapping.
    | When the offset is not None the size must also be defined.
    | The default value is None and it means the beginning of the buffer.

**discard**
    | A boolean to enable the ``GL_MAP_INVALIDATE_RANGE_BIT``
    | When this flag is True the contents of the buffer is undefined.
    | The default value is False.

.. py:method:: Buffer.unmap()

    Unmap the buffer.

.. py:attribute:: Buffer.size

    An int representing the size of the buffer in bytes.

Image
-----

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

.. py:method:: Context.image(size, format, data, samples, array, texture, cubemap) -> Image

**size**
    | The image size as a tuple of two ints.

**format**
    | The image format represented as string. (:ref:`list of image format<Image Formats>`)
    | The two most common are ``'rgba8unorm'`` and ``'depth24plus'``

**data**
    | The image content represented as ``bytes`` or a buffer for example a numpy array.
    | If the data is None the content of the image will be uninitialized. The default value is None.

**samples**
    | The number of samples for the image. Multisample render targets must have samples > 1.
    | Textures must have samples = 1. Only a power of two is possible. The default value is 1.
    | For multisampled rendering usually 4 is a good choice.

**array**
    | The number of array layers for the image. For non-array textures the value must be 0.
    | The default value is 0.

**texture**
    | A boolean representing the image to be sampled from shaders or not.
    | For textures this flag must be True, for render targets it should be False.
    | Multisampled textures to be sampled from the shaders are not supported.
    | The default is None and it means to be determined from the image type.

**cubemap**
    | A boolean representing the image to be a cubemap texture. The default value is False.

.. py:method:: Image.blit(target, target_viewport, source_viewport, filter, srgb)

**target**
    | The target image to copy to. The default value is None and it means to copy to the screen.

**target_viewport** and **source_viewport**
    | The source and target viewports defined as tuples of four ints in (x, y, width, height) format.

**filter**
    | A boolean to enable linear filtering for scaled images. By default it is True.
      It has no effect if the source and target viewports have the same size.

**srgb**
    | A boolean to enable linear to srgb conversion. By default it is False.

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
    | The size and offset defining a sub-part of the image to be read.
    | Both the size and offset are tuples of two ints.
    | The size is mandatory when the offset is not None.
    | By default the size is None and it means the full size of the image.
    | By default the offset is None and it means zero offset.

.. py:attribute:: Image.clear_value

| The clear value for the image used by the :py:meth:`Image.clear`
| For the color and stencil components the default value is zero. For depth the default value is 1.0
| For single component images the value is float or int depending on the image type.
| For multi component images the value is a tuple of ints or floats.
| The clear value type for the ``depth24plus-stencil8`` format is a tuple of float and int.

.. py:attribute:: Image.size

| The image size as a tuple of two ints.

.. py:attribute:: Image.samples

| The number of samples the image has.

.. py:attribute:: Image.color

| A boolean representing if the image is a color image.
| For depth and stencil images this value is False.

Pipeline
--------

.. py:method:: Context.pipeline(vertex_shader, fragment_shader, layout, resources, depth, stencil, blending, polygon_offset, color_mask, framebuffer, vertex_buffers, index_buffer, short_index, primitive_restart, front_face, cull_face, topology, vertex_count, instance_count, first_vertex, line_width, viewport) -> Pipeline

**vertex_shader**
    | The vertex shader code.

**fragment_shader**
    | The fragment shader code.

**layout**
    | Layout binding definition for the uniform buffers and samplers.

**resources**
    | The list of uniform buffers and samplers to be bound.

**depth**
    | The depth settings

**stencil**
    | The stencil settings

**blending**
    | The blending settings

**polygon_offset**
    | The polygon offset

**color_mask**
    | The color mask defined as a single integer.
    | The bits of the color mask grouped in fours represent the color mask for the attachments.
    | The bits in the group of fours represent the mask for the red, green, blue and alpha channels.
    | It is easier to understand it from the `implementation <https://github.com/szabolcsdombi/zengl/search?l=C%2B%2B&q=color_mask>`_.

**framebuffer**
    | A list of images representing the framebuffer for the rendering.
    | The depth or stencil attachment must be the last one in the list.
    | The size and number of samples of the images must match.

**vertex_buffers**
    | A list of vertex attribute bindings with the following keys:

        | **buffer:** A buffer to be used as the vertex attribute source
        | **format:** The vertex attribute format. (:ref:`list of vertex format<Vertex Formats>`)
        | **location:** The vertex attribute location
        | **offset:** The buffer offset in bytes
        | **stride:** The stride in bytes
        | **step:** ``'vertex'`` for per vertex attributes. ``'instance'`` for per instance attributes

    The :py:meth:`zengl.bind` method produces this list in a more compact form.

**index_buffer**
    | A buffer object to be used as the index buffer.
    | The default value is None and it means to disable indexed rendering.

**short_index**
    | A boolean to enable ``GL_UNSIGNED_SHORT`` as the index type.
    | When this flag is False the ``GL_UNSIGNED_INT`` is used.
    | The default value is False.

**primitive_restart**
    | A boolean to enable the primitive restart index. The default primitive restart index is -1.
    | The default value is True.

**front_face**
    | A string representing the front face. It must be ``'cw'`` or ``'ccw'``
    | The default value is ``'ccw'``

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

**line_width**
    | A float defining the rasterized line size in pixels. Beware wide lines are not a core feature.
    | Wondering where the point_size is? ZenGL only supports the more generic gl_PointSize_.

.. _gl_PointSize: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_PointSize.xhtml

**viewport**
    | The render viewport defined as tuples of four ints in (x, y, width, height) format.
    | The default is the full size of the framebuffer.

.. py:attribute:: Pipeline.vertex_count

    | The number of vertices or the number of elements to draw.

.. py:attribute:: Pipeline.instance_count

    | The number of instances to draw.

.. py:attribute:: Pipeline.first_vertex

    | The first vertex or the first index to start drawing from.

.. py:attribute:: Pipeline.viewport

    | The render viewport defined as tuples of four ints in (x, y, width, height) format.

Rendering to Texture
--------------------

Rendering to texture is supported. However for multisampled images must be downsampled before used as textures.
In that case an intermediate render target must be samples > 1 and texture = False.
Then this image can be downsampled with :py:meth:`Image.blit` to another image with samples = 1 and texture = True.

Shader Code
-----------

- **do** use ``#version 330`` as the first line in the shader.
- **do** use ``layout (std140)`` for uniform buffers.
- **do** use ``layout (location = ...)`` for the vertex shader inputs.
- **do** use ``layout (location = ...)`` for the fragment shader outputs.

- **don't** use ``layout (location = ...)`` for the vertex shader outputs or the fragment shader inputs.
  Matching name and order is sufficient and much more readable.

- **don't** use ``layout (binding = ...)`` for the uniform buffers or samplers.
  It is not a core feature in OpenGL 3.3 and ZenGL enforces the program layout from the pipeline parameters.

- **do** use uniform buffers, use a single one if possible.
- **don't** use uniforms, use uniform buffers instead.
- **don't** put constants in uniform buffers, use ``#include`` and string formatting.
- **don't** over-use the ``#include`` statement.
- **do** use includes without extensions.

- **do** arrange piplines in such an order to minimize framebuffer then program changes.

Shader Includes
---------------

| Shader includes were designed to solve a single problem of sharing code among shaders without having to field format the shader code.
| Includes are simple string replacements from :py:attr:`Context.includes`
| The include statement stands for including constants, functions, logic or behavior, but not files. Hence the naming should not contain extensions like ``.h``
| Nested includes do not work, they are overcomplicated and could cause other sort of issues.

**Example**

.. code-block::

    ctx.includes['common'] = '...'

    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330

            #include "common"
            #include "qtransform"

            void main() {
            }
        ''',
    )

Include Patterns
----------------

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
        return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[]({v});'

    ctx.includes['kernel'] = kernel(19)

Cleanup
-------

Clean only if necessary. It is ok not to cleanup before the program ends.

.. py:method:: Context.clear_shader_cache()

This method calls glDeleteShader for all the previously created vertex and fragment shader modules.
The resources released by this method are likely to be insignificant in size.

.. py:method:: Context.release(obj: Buffer | Image | Pipeline)

This method releases the OpenGL resources associated with the parameter.
OpenGL resources are not released automatically on grabage collection.
Release Pipelines before the Images and Buffers they use.

Utils
-----

.. py:method:: zengl.camera(eye, target, up, fov, aspect, near, far, size, clip) -> bytes

| Returns a Model-View-Projection matrix for uniform buffers.
| The return value is bytes and can be used as a parameter for :py:meth:`Buffer.write`.

.. code-block::

    mvp = zengl.camera(eye=(4.0, 3.0, 2.0), target=(0.0, 0.0, 0.0), aspect=16.0 / 9.0, fov=45.0)

.. py:method:: zengl.rgba(data: bytes, format: str) -> bytes

| Converts the image stored in data with the given format into rgba.

.. py:method:: zengl.pack(*values: Iterable[float | int]) -> bytes

| Encodes floats and ints into bytes.

.. py:method:: zengl.bind(buffer: Buffer, layout: str, *attributes: Iterable[int]) -> List[VertexBufferBinding]

| Helper function for binding a single buffer to multiple vertex attributes.
| The -1 is a special value allowed in the attributes to represent not yet implemented attribute.
| An ending ``/i`` is allowed in the layout to represent per instance stepping.

.. py:method:: zengl.calcsize(layout: str) -> int

| Calculates the size of a vertex attribute buffer layout.

.. _Image Formats:

Image Formats
-------------

==================== =================
format               OpenGL equivalent
==================== =================
r8unorm              .
rg8unorm             .
rgba8unorm           .
bgra8unorm           .
r8snorm              .
rg8snorm             .
rgba8snorm           .
r8uint               .
rg8uint              .
rgba8uint            .
r16uint              .
rg16uint             .
rgba16uint           .
r32uint              .
rg32uint             .
rgba32uint           .
r8sint               .
rg8sint              .
rgba8sint            .
r16sint              .
rg16sint             .
rgba16sint           .
r32sint              .
rg32sint             .
rgba32sint           .
r16float             .
rg16float            .
rgba16float          .
r32float             .
rg32float            .
rgba32float          .
rgba8unorm-srgb      .
bgra8unorm-srgb      .
stencil8             .
depth16unorm         .
depth24plus          .
depth24plus-stencil8 .
depth32float         .
==================== =================

.. _Vertex Formats:

Vertex Formats
--------------

========= ============= =================
shorthand vertex format OpenGL equivalent
========= ============= =================
1f        float32       .
2f        float32x2     .
3f        float32x3     .
4f        float32x4     .
1u        uint32        .
2u        uint32x2      .
3u        uint32x3      .
4u        uint32x4      .
1i        sint32        .
2i        sint32x2      .
3i        sint32x3      .
4i        sint32x4      .
2u1       uint8x2       .
4u1       uint8x4       .
2i1       sint8x2       .
4i1       sint8x4       .
2h        float16x2     .
4h        float16x4     .
2nu1      unorm8x2      .
4nu1      unorm8x4      .
2ni1      snorm8x2      .
4ni1      snorm8x4      .
2u2       uint16x2      .
4u2       uint16x4      .
2i2       sint16x2      .
4i2       sint16x4      .
2nu2      unorm16x2     .
4nu2      unorm16x4     .
2ni2      snorm16x2     .
4ni2      snorm16x4     .
========= ============= =================
