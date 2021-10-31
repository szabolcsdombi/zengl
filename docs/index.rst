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
- **simple** - *buffers, images, renderes and there you go*
- **easy-to-learn** - *it is simply OpenGL with no magic added*
- **verbose** - *most common mistakes are catched and reported in a clear and understandable way*
- **robust** - *there is no global state or external trouble-maker affecting the render*
- **backward-compatible** - *it requires OpenGL 3.3 - it is just enough*
- **cached** - *most OpenGL objects are reused between renders*
- **zen** - *there is one way to do it*

.. warning::

    This document is still in draft

.. py:class:: Instance

| Represents an OpenGL context.

.. py:class:: Buffer

| Represents an OpenGL buffer.

.. py:class:: Image

| Represents an OpenGL texture or renderbuffer.

.. py:class:: Pipeline

| Represents an entire rendering pipeline including the global state, shader program, framebuffer, vertex state,
  uniform buffer bindings, samplers and sampler bindings.

Instance
--------

.. py:method:: zengl.instance(context: Context) -> Instance

All interactions with OpenGL is done by an Instance object.
There should be a single Instance created per application.
An Instance is created with the help of a context loader.
A context loader is an object implementing the load method to resolve OpenGL functions by name.
This enables zengl to be entirely platform independent.

.. py:method:: zengl.context(headless: bool = False) -> Context

This method provides a default context loader. It requires `glcontext` to be installed.
ZenGL does not implement OpenGL function loading. glcontext is used when no alternatives are provided.

.. note::

    Implementing a context loader enables zengl to run in custom environments.
    ZenGL uses a subset of the OpenGL 3.3 core, the list of methods can be found in the project source.

**Instance for a window**

.. code-block::

    ctx = zengl.instance(zengl.context())

**Instance for headless rendering**

.. code-block::

    ctx = zengl.instance(zengl.context(headless=True))

Buffer
------

.. py:method:: Instance.buffer(data, size, dynamic) -> Buffer

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

.. py:method:: Instance.image(size, format, data, samples, texture) -> Image

Pipeline
--------

.. py:method:: Instance.pipeline(vertex_shader, fragment_shader, layout, resources, depth, stencil, blending, polygon_offset, color_mask, framebuffer, vertex_buffers, index_buffer, short_index, primitive_restart, front_face, cull_face, topology, vertex_count, instance_count, first_vertex, line_width, viewport) -> Pipeline

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

Cleanup
-------

Clean only if necessary. It is ok not to cleanup before the program ends.

.. py:method:: Instance.clear_shader_cache()

This method calls glDeleteShader for all the previously created vertex and fragment shader modules.
The resources released by this method are likely to be insignificant in size.

.. py:method:: Instance.release(obj: Buffer | Image | Pipeline)

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
