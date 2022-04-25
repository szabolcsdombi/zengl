from itertools import cycle

import imageio
import numpy as np
import zengl

import assets
from window import Window

reader = imageio.get_reader(assets.get('bunny.mp4'))  # https://test-videos.co.uk/bigbuckbunny/mp4-h264
it = cycle(reader)

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')
frame = np.zeros((720, 1280, 4), 'u1')

while window.update():
    frame[::-1, :, 0:3] = next(it)
    image.write(frame)
    image.blit()
