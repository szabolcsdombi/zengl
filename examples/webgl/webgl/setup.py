from setuptools import Extension, setup

ext = Extension(
    name='webgl',
    sources=['./webgl.c'],
)

setup(
    name='webgl',
    version='0.1.0',
    ext_modules=[ext],
)
