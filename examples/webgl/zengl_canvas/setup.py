from setuptools import Extension, setup

ext = Extension(
    name='zengl_canvas',
    sources=['./zengl_canvas.c'],
)

setup(
    name='zengl_canvas',
    version='0.1.0',
    ext_modules=[ext],
)
