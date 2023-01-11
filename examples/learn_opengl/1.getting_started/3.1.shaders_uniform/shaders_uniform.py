import struct

import numpy as np
import zengl

from utils import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.3, 0.3, 0.3, 1.0)

vertex_buffer = ctx.buffer(np.array([
    0.5, -0.5, 0.0,   # bottom right
    -0.5, -0.5, 0.0,  # bottom left
    0.0, 0.5, 0.0,    # top
], 'f4'))

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
        uniform vec4 ourColor;

        void main()
        {
            FragColor = ourColor;
        }
    ''',
    # Uniforms defined this way will be set by the pipeline.render() call.
    # It is possible to mutate these values.
    # To share uniforms across multiple pipeline objects using uniform buffers is the way to go.
    # For simplicity we will use uniforms.
    uniforms={
        'ourColor': (0.0, 0.0, 0.0, 1.0),
    },
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f', 0),
    vertex_count=3,
)


while window.update():
    image.clear()
    depth.clear()

    # The pipeline.uniforms is dictionary with uniform names mapped to memoryview objects.
    # With ZenGL uniforms are rarely used. With the exception of pipeline specific flags the uniforms
    # should go into a single uniform buffer bound to all pipeline objects.
    # Updating uniforms with ZenGL may look odd. The pipeline object holds a single C/C++ buffer for the uniforms.
    # This buffer is exposed by the pipeline.uniforms. Writing to a memoryview is done as follows.
    green = np.sin(window.time) / 2.0 + 0.5
    pipeline.uniforms['ourColor'][:] = struct.pack('4f', 0.0, green, 0.0, 1.0)

    pipeline.render()
    image.blit()
