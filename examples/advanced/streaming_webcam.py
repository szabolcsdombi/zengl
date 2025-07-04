import sys

import cv2
import numpy as np
import pygame
import zengl
import zengl_extras

cap = cv2.VideoCapture(index=1)  # (try different indices if 1 doesn't work)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

ret, cv_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to grab initial frame from webcam")

height, width = cv_frame.shape[:2]
window_size = (width, height)

zengl_extras.init()

pygame.init()
pygame.display.set_mode(window_size, flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm')

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    if not ret:
        continue

    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    flipped_frame = np.flip(rgba, axis=(0, 1))
    image.write(flipped_frame)
    image.blit()

    pygame.display.flip()
