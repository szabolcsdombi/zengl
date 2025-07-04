import sys
from itertools import cycle

import assets
import imageio
import numpy as np
import pygame
import zengl
import zengl_extras

reader = imageio.get_reader(assets.get('bunny.mp4'))  # https://test-videos.co.uk/bigbuckbunny/mp4-h264
it = cycle(reader)

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)
window_size = pygame.display.get_window_size()

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm')
frame = np.zeros((720, 1280, 4), 'u1')

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    frame[::-1, :, 0:3] = next(it)
    image.write(frame)
    image.blit()

    pygame.display.flip()
