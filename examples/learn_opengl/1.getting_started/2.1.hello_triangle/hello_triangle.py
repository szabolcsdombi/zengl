import numpy as np
import zengl

from utils import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.3, 0.3, 0.3, 1.0)

# We can create an ZenGL buffer with initial content by passing a numpy array as the first parameter.
# The dtype is "f4" for 32bit floats. The numpy array implements the buffer interface.
# We could have passed np.array([], 'f4').tobytes() as well.
vertex_buffer = ctx.buffer(np.array([
    -0.5, -0.5, 0.0,  # let
    0.5, -0.5, 0.0,   # right
    0.0, 0.5, 0.0,    # top
], 'f4'))

# In ZenGL all the rendering is encapsulated in a single Pipeline object.
# The state required for the render is passed as parameters at creation time.
pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core
        layout (location = 0) in vec3 aPos;
        void main()
        {
            gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
        }
    ''',

    # In ZenGL the framebuffer is a simple list of ZenGL images.
    # Depth or stencil must be the last item.
    framebuffer=[image, depth],

    # Rendering mode is GL_TRIANGLES
    topology='triangles',

    # Bind vertex_buffer to attribute 0 as vec3.
    vertex_buffers=zengl.bind(vertex_buffer, '3f', 0),
    vertex_count=3,
)

while window.update():
    image.clear()
    depth.clear()

    # After the images are cleared we invoke our rendering pipeline.
    # The pipeline.render() takes no arguments by design.
    pipeline.render()
    image.blit()
