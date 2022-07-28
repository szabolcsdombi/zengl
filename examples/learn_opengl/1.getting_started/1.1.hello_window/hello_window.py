import zengl

from utils import Window

# We create a pyglet window with OpenGL support.
window = Window()

# The window provides an OpenGL context to which zengl can connect to.
# ZenGL loads the necessary OpenGL functions.
ctx = zengl.context()

# ZenGL can only render to user defined images.
# We can create our default framebuffer with a color and depth attachment.
image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)

# Use gray as clear color
image.clear_value = (0.3, 0.3, 0.3, 1.0)

# Initialization goes here

# In these examples the main loop will be controlled by the user.
while window.update():
    # Per frame logic goes here.

    # Clear the color and depth images.
    image.clear()
    depth.clear()

    # Blit the color image content to the screen.
    image.blit()
