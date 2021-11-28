import sys

from setuptools import Extension, setup

extra_compile_args = []

if sys.platform.startswith('linux'):
    extra_compile_args = ['-fpermissive', '-Wno-write-strings', '-Wno-narrowing']

if sys.platform.startswith('darwin'):
    extra_compile_args = ['-std=c++11', '-Wno-writable-strings', '-Wno-c++11-narrowing']

ext = Extension(
    name='zengl',
    sources=['zengl.cpp'],
    depends=['zengl.hpp'],
    extra_compile_args=extra_compile_args,
)

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='zengl',
    version='1.2.0',
    py_modules=['_zengl'],
    data_files=[('.', ['zengl.pyi'])],
    ext_modules=[ext],
    description='high-performance rendering',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/szabolcsdombi/zengl/',
    author='Szabolcs Dombi',
    author_email='cprogrammer1994@gmail.com',
    license='MIT',
    extras_require={
        'examples': [
            'ffmpeg',
            'glcontext',
            'imageio-ffmpeg',
            'imageio',
            'numpy',
            'objloader',
            'pillow',
            'pybullet',
            'pygame',
            'pyglet',
            'pygmsh',
            'scikit-image',
        ],
    },
)
