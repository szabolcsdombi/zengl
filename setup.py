import os
import sys

from setuptools import Extension, setup

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith('linux'):
    extra_compile_args = ['-fpermissive', '-Wno-write-strings', '-Wno-narrowing']

if sys.platform.startswith('darwin'):
    extra_compile_args = ['-std=c++11', '-Wno-writable-strings', '-Wno-c++11-narrowing']

if os.getenv('ZENGL_COVERAGE'):
    extra_compile_args += ['-O0', '--coverage']
    extra_link_args += ['-O0', '--coverage']

ext = Extension(
    name='zengl',
    sources=['zengl.cpp'],
    depends=['zengl.hpp'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='zengl',
    version='2.0.0a2',
    ext_modules=[ext],
    py_modules=['_zengl'],
    packages=['zengl-stubs'],
    package_data={'zengl-stubs': ['__init__.pyi']},
    include_package_data=True,
    license='MIT',
    python_requires='>=3.6',
    platforms=['any'],
    description='Compact Python OpenGL rendering library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Szabolcs Dombi',
    author_email='cprogrammer1994@gmail.com',
    url='https://github.com/szabolcsdombi/zengl/',
    project_urls={
        'Documentation': 'https://zengl.readthedocs.io/',
        'Source': 'https://github.com/szabolcsdombi/zengl/',
        'Bug Tracker': 'https://github.com/szabolcsdombi/zengl/issues/',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Games/Entertainment',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords=[
        'OpenGL',
        'rendering',
        'graphics',
        'visualization',
    ],
    install_requires=['glcontext'],
    extras_require={
        'examples': [
            'chull',
            'ffmpeg-python',
            'glcontext',
            'imageio-ffmpeg',
            'imageio',
            'matplotlib',
            'numpy',
            'objloader',
            'pillow',
            'progress',
            'pybullet',
            'pygame',
            'pyglet',
            'pygmsh',
            'pyopengl',
            'requests',
            'scikit-image',
            'vmath',
        ],
    },
)
