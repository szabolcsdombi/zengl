from setuptools import Extension, setup

ext = Extension(
    name='zengl_webgl',
    sources=['./zengl_webgl.c'],
)

setup(
    name='zengl_webgl',
    version='0.1.0',
    ext_modules=[ext],
)
