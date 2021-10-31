# ZenGL

ZenGL is a minimalist Python module providing exactly **one** way to render scenes with OpenGL.

```
pip install zengl
```

- [Documentation](https://zengl.readthedocs.io/)
- [zengl on Github](https://github.com/szabolcsdombi/zengl/)
- [zengl on PyPI](https://pypi.org/project/zengl/)

**ZenGL is ...**

- **high-performance**
- **simple** - *buffers, images, pipelines and there you go*
- **easy-to-learn** - *it is simply OpenGL with no magic added*
- **verbose** - *most common mistakes are catched and reported in a clear and understandable way*
- **robust** - *there is no global state or external trouble-maker affecting the render*
- **backward-compatible** - *it requires OpenGL 3.3 - it is just enough*
- **cached** - *most OpenGL objects are reused between renders*
- **zen** - *there is one way to do it*

## Concept

ZenGL provides a simple way to render from Python. We aim to support headless rendering first,
rendering to a window is done by blitting the final image to the screen. By doing this we have full control of
what we render. The window does not have to be multisample, and it requires no depth buffer at all.

Offscreen rendering works out of the box on all platforms if the right loader is provided.
Loaders implement a load method to resolve a subset of OpenGL 3.3 core. The return value of the the load method is
an int, a void pointer to the function implementation.
Virtualized, traced and debug environments can be provided by custom loaders.
The current implementation uses the glcontext from moderngl to load the OpenGL methods.

ZenGL's main focus is on readability and maintainability. Pipelines in ZenGL are almost entirely immutable and they
cannot affect each other expect when one draws on top of the other's result that is totally expected.
No global state is affecting the render, if something breaks there is one place to debug.

ZenGL does not use anything beyond OpenGL 3.3 core, not even if the more convenient methods are available.
Implementation is kept simple. Usually this is not a bottleneck.

ZenGL does not implement transform feedback, storage buffers or storage images, tesselation, geometry shader and maybe many more.
We have a strong reason not to include them in the feature list. They add to complexity and are against ZenGL's main philosophy.
ZenGL was built on top experience gathered on real life projects that could never make good use of any of that.

ZenGL is using the same vertex and image format naming as WebGPU and keeping the vertex array definition from ModernGL.
ZenGL is not the next version of ModernGL. ZenGL is a simplification of a subset of ModernGL with some extras
that was not possible to include in ModernGL.

Future plans include a Vulkan 1.0 backend that was considered when designing and developing ZenGL.

## Example

```py
import zengl
from PIL import Image

ctx = zengl.context(zengl.loader(headless=True))

size = (1280, 720)
image = ctx.image(size, 'rgba8unorm', samples=1)

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330

        out vec3 v_color;

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

image.clear_value = (1.0, 1.0, 1.0, 1.0)
image.clear()
triangle.render()

Image.frombuffer('RGBA', size, image.read(), 'raw', 'RGBA', 0, -1).save('hello.png')
```
