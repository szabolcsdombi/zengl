ZenGL
=====

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

Objects
=======

.. py:class:: Instance

| Represents an OpenGL context.

.. py:class:: Buffer

| Represents an OpenGL buffer.

.. py:class:: Image

| Represents an OpenGL texture or renderbuffer.

.. py:class:: Renderer

| Represents an entire rendering pipeline including the global state, shader progra, framebuffer, vertex state,
| uniform buffer bindings, samplers and sampler bindings.

Documentation
=============

Instance Objects
----------------

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

Buffer Objects
--------------

.. py:method:: Instance.buffer(data, size, dynamic) -> Buffer

Buffer objects hold data used by rendering.
Buffers are not variable sized, they are allocated upfront in the device memory.

.. code-block::

    vertex_buffer = ctx.buffer(np.array([0.0, 0.0, 1.0, 1.0], 'f4'))

.. code-block::

    index_buffer = ctx.buffer(np.array([0, 1, 2], 'i4'))

.. code-block::

    vertex_buffer = ctx.buffer(size=1024)

Image Objects
-------------

.. py:method:: Instance.image(size, format, data, samples, texture) -> Image

Renderer Objects
----------------

.. py:method:: Instance.renderer(vertex_shader, fragment_shader, layout, resources, depth, stencil, blending, polygon_offset, color_mask, framebuffer, vertex_buffers, index_buffer, short_index, primitive_restart, front_face, cull_face, topology, vertex_count, instance_count, first_vertex, line_width, viewport) -> Renderer
