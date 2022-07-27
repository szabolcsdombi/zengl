import imageio
import numpy as np
import zengl

from window import Window

reader = imageio.get_reader('<video0>')
it = iter(reader)
height, width = next(it).shape[:2]
frame = np.zeros((height, width, 4), 'u1')

window = Window((width, height))
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')

while window.update():
    frame[::-1, ::-1, 0:3] = next(it)
    image.write(frame)
    image.blit()
