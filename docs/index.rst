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

.. py:method:: zengl.loader(headless: bool = False) -> Context

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

.. py:method:: Context.buffer(data, size, dynamic) -> Buffer

Buffer objects hold data used by rendering.
Buffers are not variable sized, they are allocated upfront in the device memory.

.. code-block::

    vertex_buffer = ctx.buffer(np.array([0.0, 0.0, 1.0, 1.0], 'f4'))

.. code-block::

    index_buffer = ctx.buffer(np.array([0, 1, 2], 'i4'))

.. code-block::

    vertex_buffer = ctx.buffer(size=1024)

Image
-----

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

.. py:method:: Image.blit(target, target, target_viewport, source_viewport, filter, srgb)

**target**
    | The target image to copy to. The default value is None and it means to copy to the screen.

**target_viewport** and **source_viewport**
    | The source and target viewports defined as tuples of four ints in (x, y, width, height) format.

**filter**
    | A boolean to enable linear filtering for scaled images. By default it is True.
      It has no effect if the source and target viewports have the same size.

**srgb**
    | A boolean to enable linear to srgb conversion. By default it is False.

Pipeline
--------

.. py:method:: Context.pipeline(vertex_shader, fragment_shader, layout, resources, depth, stencil, blending, polygon_offset, color_mask, framebuffer, vertex_buffers, index_buffer, short_index, primitive_restart, front_face, cull_face, topology, vertex_count, instance_count, first_vertex, line_width, viewport) -> Pipeline

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

| Shorthand for binding a single buffer to multiple vertex attributes.

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
